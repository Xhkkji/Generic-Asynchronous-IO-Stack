import math

import torch
from torchvision.transforms import CenterCrop, InterpolationMode, RandomResizedCrop
from torchvision.transforms.functional import resized_crop


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class ImageNetBatchPreprocessor:
    # 统一的在线图像预处理：
    # - train: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize
    # - val: CenterCrop(224) + Normalize
    def __init__(
        self,
        source_image_size,
        target_image_size=224,
        train_scale=(0.8, 1.0),
        train_ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip_prob=0.5,
    ):
        self.source_image_size = int(source_image_size)
        self.target_image_size = int(target_image_size)
        self.train_scale = tuple(float(x) for x in train_scale)
        self.train_ratio = tuple(float(x) for x in train_ratio)
        self.hflip_prob = float(hflip_prob)
        self.eval_center_crop = CenterCrop(self.target_image_size)

    def _normalize(self, images):
        mean = _IMAGENET_MEAN.to(device=images.device)
        std = _IMAGENET_STD.to(device=images.device)
        return images.sub(mean).div(std)

    def train(self, images):
        processed = []
        for image in images:
            i, j, h, w = RandomResizedCrop.get_params(
                image,
                scale=self.train_scale,
                ratio=self.train_ratio,
            )
            image = resized_crop(
                image,
                i,
                j,
                h,
                w,
                size=[self.target_image_size, self.target_image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            # 训练预处理功能：
            # - 保持逐图随机翻转行为，registered 路径目前更依赖这类隐式节流
            if self.hflip_prob > 0.0 and torch.rand(1, device=image.device).item() < self.hflip_prob:
                image = torch.flip(image, dims=[2])
            processed.append(image)
        return self._normalize(torch.stack(processed, dim=0))

    def eval(self, images):
        processed = [self.eval_center_crop(image) for image in images]
        return self._normalize(torch.stack(processed, dim=0))
