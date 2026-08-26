"""Models

FareMark: AlexNet, ShuffleNet, ResNet-18 and GoogleNet on MNIST / CIFAR-10 / CIFAR-100. 
Implemented: ResNet-18 + tiny SmallCNN for fast smoke tests 
ShuffleNet / GoogleNet are to be added later via `build_model`
Tested: ResNet-18 on CIFAR-100

NOTE: adapted for small images (28x28 / 32x32). 
torchvision ResNet-18 is built for 224x224 ImageNet inputs - took the standard adaptation of CIFAR ResNet
"""
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        net = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        # CIFAR/MNIST stem adaptation.
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


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
    if name == "resnet18":
        return ResNet18(num_classes, in_channels)
    if name == "smallcnn":
        return SmallCNN(num_classes, in_channels)
    raise ValueError(f"Unknown model '{name}'. Add it to build_model().")
