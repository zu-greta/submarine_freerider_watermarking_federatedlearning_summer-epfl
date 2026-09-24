"""Models

FareMark: AlexNet, ShuffleNet, ResNet-18 and GoogleNet on MNIST / CIFAR-10 / CIFAR-100. 
Implemented: ResNet-18 + tiny SmallCNN for fast smoke tests 
ShuffleNet / GoogleNet are to be added later via `build_model`
Tested: ResNet-18 on CIFAR-100

NOTE: adapted for small images (28x28 / 32x32).
torchvision ResNet-18 is built for 224x224 ImageNet inputs - took the standard adaptation of CIFAR ResNet

-------------------------------------------------------------------------------
NOTE - Model Architecture and scope mapping
-------------------------------------------------------------------------------
Output layer watermarking lives in the softmax layer. The scope maps in 
AdaptiveTapFreeRider / GraftBlockFreeRider (`_SCOPE_KEEP`) count *named parameter 
tensors from the end* of this exact model:

  This ResNet-18 (CIFAR stem) has 62 named parameter tensors (~11.2M scalars).
  named_parameters() order is registration order, so the last tensors are:
        ... layer4.1.conv2.weight,
            layer4.1.bn2.weight, layer4.1.bn2.bias,
            fc.weight, fc.bias
  scope "head2"  = keep 5  -> [layer4.1.conv2.weight, layer4.1.bn2.{weight,bias},
                              fc.weight, fc.bias]  ~= 2.41M scalars (~21%).
                              = the softmax/output fc + the conv right before it.
  scope "block2" = keep 20 -> last ~2.5 residual blocks + fc, ~9.04M scalars (~80%).
  scope "head"   = keep 2  -> [fc.weight, fc.bias] only.
  scope "full"   = keep all (identical to the honest path).

If you swap in a different backbone, re-derive `_SCOPE_KEEP` for it
- the head2/block2 offsets are model-specific.
"""
import os
import torch.nn as nn
import torchvision


class ResNetX(nn.Module):
    """torchvision ResNet-18/34/50 

    cifar_stem=True  -> 3x3 stride-1 conv1 + maxpool=Identity (32px)
    cifar_stem=False -> stock ImageNet stem (7x7 s2 conv1 + 3x3 maxpool)
    """
    _CTOR = {"resnet18": torchvision.models.resnet18,
             "resnet34": torchvision.models.resnet34,
             "resnet50": torchvision.models.resnet50}

    def __init__(self, arch: str, num_classes: int, in_channels: int,
                 cifar_stem: bool = True):
        super().__init__()
        net = self._CTOR[arch](weights=None, num_classes=num_classes)
        if cifar_stem:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
            net.maxpool = nn.Identity()
        elif in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        self.net = net

    def forward(self, x):
        return self.net(x)


# Back-compat: ResNet18(num_classes, in_channels) == CIFAR-stem resnet18.
class ResNet18(ResNetX):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__("resnet18", num_classes, in_channels, cifar_stem=True)


def _use_cifar_stem() -> bool:
    """CIFAR stem for small 32px datasets"""
    stem = os.environ.get("RESNET_STEM", "").strip().lower()
    if stem in ("cifar", "imagenet"):
        return stem == "cifar"
    return os.environ.get("DATASET", "").strip().lower() not in ("food101",)


class SmallCNN(nn.Module):
    """Tiny net for fast pipeline smoke tests (a few rounds, high MNIST acc)."""

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def build_model(name: str, num_classes: int, in_channels: int) -> nn.Module:
    name = name.lower()
    if name in ("resnet18", "resnet34", "resnet50"):
        return ResNetX(name, num_classes, in_channels, cifar_stem=_use_cifar_stem())
    if name == "smallcnn":
        return SmallCNN(num_classes, in_channels)
    raise ValueError(f"Unknown model '{name}'. Add it to build_model().")