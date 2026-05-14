import torch
import torch.nn as nn


class ResNet50Classifier(nn.Module):
    # 最小版 ResNet-50 分类网络：
    # 默认输入是 3 通道图片，输出为 num_classes 类。
    def __init__(self, num_classes, pretrained=False, in_channels=3):
        super().__init__()

        try:
            from torchvision.models import resnet50, ResNet50_Weights
        except ImportError as exc:
            raise ImportError("使用 ResNet50Classifier 需要 torchvision") from exc

        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)

        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)


def build_resnet50(num_classes, pretrained=False, in_channels=3):
    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    model = build_resnet50(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("output shape:", tuple(y.shape))
