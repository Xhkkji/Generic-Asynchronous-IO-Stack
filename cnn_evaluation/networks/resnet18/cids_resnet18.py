import torch
import torch.nn as nn


class ResNet18Classifier(nn.Module):
    # 最小版 ResNet18 分类网络：
    # 默认输入是 3 通道图片，输出为 num_classes 类。
    def __init__(self, num_classes, pretrained=False, in_channels=3):
        super().__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError as exc:
            raise ImportError("使用 ResNet18Classifier 需要 torchvision") from exc

        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = resnet18(weights=None)

        # 如果输入通道不是 3，则替换第一层卷积。
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
        # 输入形状: [B, C, H, W]
        return self.backbone(images)


def build_resnet18(num_classes, pretrained=False, in_channels=3):
    # 便于训练脚本直接构建模型的 helper。
    return ResNet18Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    model = build_resnet18(num_classes=200)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print("output shape:", tuple(y.shape))
