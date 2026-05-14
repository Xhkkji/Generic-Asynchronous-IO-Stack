import torch
import torch.nn as nn


class AlexNetClassifier(nn.Module):
    # 最小版 AlexNet 分类网络：
    # 默认输入是 3 通道图片，输出为 num_classes 类。
    def __init__(self, num_classes, pretrained=False, in_channels=3):
        super().__init__()

        try:
            from torchvision.models import alexnet, AlexNet_Weights
        except ImportError as exc:
            raise ImportError("使用 AlexNetClassifier 需要 torchvision") from exc

        if pretrained:
            self.backbone = alexnet(weights=AlexNet_Weights.DEFAULT)
        else:
            self.backbone = alexnet(weights=None)

        if in_channels != 3:
            first_conv = self.backbone.features[0]
            self.backbone.features[0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)


def build_alexnet(num_classes, pretrained=False, in_channels=3):
    return AlexNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    model = build_alexnet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("output shape:", tuple(y.shape))
