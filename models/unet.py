import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.models import mobilenet_v2


class Conv2DBnReLu(nn.Layer):
    def __init__(
        self, 
        in_chnnels, out_channels, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1):
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2D(
                in_chnnels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

class MobileNetV2(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        mobilenet = mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features

    def forward(self, x):
        # stage 1
        feat1 = self.features._sub_layers['0'](x)  # [N, 32, 128, 128]

        # stage 2
        feat2 = feat1
        for key in ['1', '2', '3']:
            feat2 = self.features[key](feat2)  # [N, 24, 64, 64]

        # stage 3
        feat3 = feat2
        for key in ['4', '5', '6']:
            feat3 = self.features[key](feat3)  # [N, 32, 32, 32]

        # stage 4
        feat4 = feat3
        for key in ['7', '8', '9', '10']:
            feat4 = self.features[key](feat4)  # [N, 64, 16, 16]

        # stage 5
        feat5 = feat4
        for key in ['11', '12', '13', '14', '15', '16', '17', '18']:
            feat5 = self.features[key](feat5)   # [N, 1280, 8, 8]

        return feat1, feat2, feat3, feat4, feat5
    
    
class UNet(nn.Layer):
    def __init__(self, n_classes=10, bb_pretrained=True, return_feats=False):
        super().__init__()
        
        self.n_classes = n_classes
        self.return_feats = return_feats
        self.mbv2_bb = MobileNetV2(bb_pretrained)
        
        self.reduction = Conv2DBnReLu(1280, 64, 1)
        
        self.d1 = nn.Sequential(
            Conv2DBnReLu(128, 64, 3, 1, 1), Conv2DBnReLu(64, 32, 3, 1, 1)
        )
        self.d2 = nn.Sequential(
            Conv2DBnReLu(64, 64, 3, 1, 1), Conv2DBnReLu(64, 24, 3, 1, 1)
        )
        self.d3 = nn.Sequential(
            Conv2DBnReLu(48, 48, 3, 1, 1), Conv2DBnReLu(48, 32, 3, 1, 1)
        )
        self.d4 = nn.Sequential(
            Conv2DBnReLu(64, 32, 3, 1, 1), Conv2DBnReLu(32, 32, 3, 1, 1)
        )
        
        self.classifer = nn.Conv2D(32, self.n_classes, 3, 1, 1)
        
        
    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.mbv2_bb(x)
        feat5_rd = self.reduction(feat5)
        
        out = self.d1(paddle.concat([F.upsample(feat5_rd, scale_factor=2, mode='bilinear'), feat4], axis=1))
        out = self.d2(paddle.concat([F.upsample(out, scale_factor=2, mode='bilinear'), feat3], axis=1))
        out = self.d3(paddle.concat([F.upsample(out, scale_factor=2, mode='bilinear'), feat2], axis=1))
        out = self.d4(paddle.concat([F.upsample(out, scale_factor=2, mode='bilinear'), feat1], axis=1))
        
        logits = self.classifer(F.upsample(out, scale_factor=2, mode='bilinear'))
        
        if self.return_feats:
            return logits, (feat1, feat2, feat3, feat4, feat5)
        return logits





    
    