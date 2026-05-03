"""ResNet-18 and ResNet-50 for CIFAR-10 Knowledge Distillation.

Both models expose intermediate feature maps for Feature-KD via
forward hooks stored in self.features after each forward pass.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet(nn.Module):
    """Wrapper around torchvision ResNet with CIFAR-10 head and feature hooks.

    Args:
        variant:   'resnet18' (student) or 'resnet50' (teacher).
        num_classes: Number of output classes (default: 10 for CIFAR-10).
        pretrained:  Load ImageNet pretrained weights.
    """

    def __init__(
        self,
        variant: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = False,
    ):
        super().__init__()

        if variant == "resnet18":
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif variant == "resnet50":
            base = tv_models.resnet50(
                weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        # Replace final FC for CIFAR-10
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.model = base
        self.features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward hooks to layer2, layer3, layer4 for Feature-KD."""
        for name in ("layer2", "layer3", "layer4"):
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet(variant: str, num_classes: int = 10, pretrained: bool = False) -> ResNet:
    return ResNet(variant=variant, num_classes=num_classes, pretrained=pretrained)