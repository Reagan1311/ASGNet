import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
from model.module.decoder import build_decoder
from model.module.ASPP import ASPP

# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        layers = args.layers
        classes = args.classes
        sync_bn = args.sync_bn
        pretrained = True
        assert layers in [50, 101, 152]
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.shot = args.shot
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.pyramid = args.pyramid

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.Dropout2d(p=0.5)
        )

        # Using Feature Enrichment Module from PFENet as context module
        if self.pyramid:

            self.pyramid_bins = args.ppm_scales
            self.avgpool_list = []

            for bin in self.pyramid_bins:
                if bin > 1:
                    self.avgpool_list.append(
                        nn.AdaptiveAvgPool2d(bin)
                    )

            self.corr_conv = []
            self.beta_conv = []
            self.inner_cls = []

            for bin in self.pyramid_bins:
                self.corr_conv.append(nn.Sequential(
                    nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                ))
                self.beta_conv.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                ))
                self.inner_cls.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1)
                ))
            self.corr_conv = nn.ModuleList(self.corr_conv)
            self.beta_conv = nn.ModuleList(self.beta_conv)
            self.inner_cls = nn.ModuleList(self.inner_cls)

            self.alpha_conv = []
            for idx in range(len(self.pyramid_bins) - 1):
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(2 * reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU(inplace=True)
                ))
            self.alpha_conv = nn.ModuleList(self.alpha_conv)

            self.res1 = nn.Sequential(
                nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.res2 = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

        # Using ASPP as context module
        else:
            self.ASPP = ASPP(out_channels=reduce_dim)
            self.corr_conv = nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
            )

            self.skip1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.skip2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.skip3 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.decoder = build_decoder(256)
            self.cls_aux = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.1),
                                         nn.Conv2d(reduce_dim, classes, kernel_size=1))

    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), s_seed=None, y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_conv(query_feat)

        # Support Feature
        supp_feat_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                mask_list.append(mask)  # Get all the downsampled mask

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_conv(supp_feat)
            supp_feat_list.append(supp_feat)  # shot x [bs x 256 x h x w]

########################### Adaptive Superpixel Clustering ###########################
        bs, _, max_num_sp, _ = s_seed.size()  # bs x shot x max_num_sp x 2

        guide_feat_list = []
        prob_map_list = []
        for bs_ in range(bs):
            sp_center_list = []
            query_feat_ = query_feat[bs_, :, :, :]  # c x h x w
            for shot_ in range(self.shot):
                with torch.no_grad():
                    supp_feat_ = supp_feat_list[shot_][bs_, :, :, :]  # c x h x w
                    supp_mask_ = mask_list[shot_][bs_, :, :, :]       # 1 x h x w
                    s_seed_ = s_seed[bs_, shot_, :, :]                # max_num_sp x 2
                    num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))

                    # if num_sp == 0 or 1, use the Masked Average Pooling instead
                    if (num_sp == 0) or (num_sp == 1):
                        supp_proto = Weighted_GAP(supp_feat_.unsqueeze(0), supp_mask_.unsqueeze(0))  # 1 x c x 1 x 1
                        sp_center_list.append(supp_proto.squeeze().unsqueeze(-1))                    # c x 1
                        continue

                    s_seed_ = s_seed_[:num_sp, :]  # num_sp x 2
                    sp_init_center = supp_feat_[:, s_seed_[:, 0], s_seed_[:, 1]]  # c x num_sp (sp_seed)
                    sp_init_center = torch.cat([sp_init_center, s_seed_.transpose(1, 0).float()], dim=0)  # (c + xy) x num_sp

                    if self.training:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=self.train_iter)
                        sp_center_list.append(sp_center)
                    else:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=self.eval_iter)
                        sp_center_list.append(sp_center)

            sp_center = torch.cat(sp_center_list, dim=1)   # c x num_sp_all (collected from all shots)

########################### Guided Prototype Allocation ###########################
            # when support only has one prototype in 1-shot training
            if (self.shot == 1) and (sp_center.size(1) == 1):
                cos_sim_map = F.cosine_similarity(sp_center[..., None], query_feat_, dim=0, eps=1e-7)  # 1 x h x w
                prob_map_list.append(cos_sim_map.unsqueeze(0).unsqueeze(0))
                sp_center_tile = sp_center[None, ..., None].expand(-1, -1, query_feat_.size(1), query_feat_.size(2))  # 1 x c x h x w
                guide_feat = torch.cat([query_feat_.unsqueeze(0), sp_center_tile], dim=1)  # 1 x 2c x h x w
                guide_feat_list.append(guide_feat)
                continue

            sp_center_rep = sp_center[..., None, None].repeat(1, 1, query_feat_.size(1), query_feat_.size(2))
            cos_sim_map = F.cosine_similarity(sp_center_rep, query_feat_.unsqueeze(1), dim=0, eps=1e-7)  # num_sp x h x w
            prob_map = cos_sim_map.sum(0, keepdim=True)  # 1 x h x w
            prob_map_list.append(prob_map.unsqueeze(0))

            guide_map = cos_sim_map.max(0)[1]  # h x w
            sp_guide_feat = sp_center[:, guide_map]  # c x h x w
            guide_feat = torch.cat([query_feat_, sp_guide_feat], dim=0)  # 2c x h x w
            guide_feat_list.append(guide_feat.unsqueeze(0))

        guide_feat = torch.cat(guide_feat_list, dim=0)  # bs x 2c x h x w
        prob_map = torch.cat(prob_map_list, dim=0)      # bs x 1 x h x w

########################### Context Module ###########################
        if self.pyramid:

            out_list = []
            pyramid_feat_list = []

            for idx, tmp_bin in enumerate(self.pyramid_bins):
                if tmp_bin <= 1.0:
                    bin = int(guide_feat.shape[2] * tmp_bin)
                    guide_feat_bin = nn.AdaptiveAvgPool2d(bin)(guide_feat)
                else:
                    bin = tmp_bin
                    guide_feat_bin = self.avgpool_list[idx](guide_feat)
                prob_map_bin = F.interpolate(prob_map, size=(bin, bin), mode='bilinear', align_corners=True)
                merge_feat_bin = torch.cat([guide_feat_bin, prob_map_bin], 1)
                merge_feat_bin = self.corr_conv[idx](merge_feat_bin)

                if idx >= 1:
                    pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                    pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                    rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                    merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

                merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
                inner_out_bin = self.inner_cls[idx](merge_feat_bin)
                merge_feat_bin = F.interpolate(merge_feat_bin, size=(guide_feat.size(2), guide_feat.size(3)),
                                               mode='bilinear', align_corners=True)
                pyramid_feat_list.append(merge_feat_bin)
                out_list.append(inner_out_bin)

            final_feat = torch.cat(pyramid_feat_list, 1)
            final_feat = self.res1(final_feat)
            final_feat = self.res2(final_feat) + final_feat
            out = self.cls(final_feat)

        # ASPP structure
        else:
            final_feat = self.corr_conv(torch.cat([guide_feat, prob_map], 1))
            final_feat = final_feat + self.skip1(final_feat)
            final_feat = final_feat + self.skip2(final_feat)
            final_feat = final_feat + self.skip3(final_feat)
            final_feat = self.ASPP(final_feat)
            decoder_out = self.decoder(final_feat, query_feat_1)
            out = self.cls(decoder_out)

        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            if self.pyramid:
                for idx_k in range(len(out_list)):
                    inner_out = out_list[idx_k]
                    inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                    aux_loss = aux_loss + self.criterion(inner_out, y.long())
                aux_loss = aux_loss / len(out_list)
            else:
                aux_out = self.cls_aux(final_feat)
                aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = self.criterion(aux_out, y.long())

            return out.max(1)[1], main_loss, aux_loss
        else:
            return out

    def sp_center_iter(self, supp_feat, supp_mask, sp_init_center, n_iter):
        '''
        :param supp_feat: A Tensor of support feature, (C, H, W)
        :param supp_mask: A Tensor of support mask, (1, H, W)
        :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
        :param n_iter: The number of iterations
        :return: sp_center: The centroid of superpixels (prototypes)
        '''

        c_xy, num_sp = sp_init_center.size()
        _, h, w = supp_feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
        supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
        supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]  # (C + xy) x num_roi

        num_roi = supp_feat_roi.size(1)
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
            sp_center = sp_center.sum(1)

        return sp_center[:-2, :]

    def _optimizer(self, args):
        if self.pyramid:
            optimizer = torch.optim.SGD(
                [
                    {'params': self.down_conv.parameters()},
                    {'params': self.corr_conv.parameters()},
                    {'params': self.alpha_conv.parameters()},
                    {'params': self.beta_conv.parameters()},
                    {'params': self.inner_cls.parameters()},
                    {'params': self.res1.parameters()},
                    {'params': self.res2.parameters()},
                    {'params': self.cls.parameters()},
                ],
                lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': self.down_conv.parameters()},
                    {'params': self.corr_conv.parameters()},
                    {'params': self.skip1.parameters()},
                    {'params': self.skip2.parameters()},
                    {'params': self.skip3.parameters()},
                    {'params': self.ASPP.parameters()},
                    {'params': self.decoder.parameters()},
                    {'params': self.cls.parameters()},
                    {'params': self.cls_aux.parameters()},
                ],
                lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer
