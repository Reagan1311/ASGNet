import torch
import torch.nn as nn
import torch.nn.functional as F


class build_decoder(nn.Module):
    def __init__(self, low_dim):
        super(build_decoder, self).__init__()

        low_dim_compress = low_dim // 4
        self.conv1 = nn.Conv2d(low_dim, low_dim_compress, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_dim_compress)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Sequential(nn.Conv2d(low_dim + low_dim_compress,
                                                 low_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(low_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
