import torch.nn as nn
from torchvision import models


# --- Helper functions for building specific model families ---

def _build_resnet(model_name, weights, num_classes):
    """Helper function to build and modify a ResNet model."""
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def _build_vit(model_name, weights, num_classes):
    """Helper function to build and modify a Vision Transformer model."""
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(weights=weights)
    else:
        raise ValueError(f"Unsupported ViT variant: {model_name}")

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    return model


def _build_efficientnet(model_name, weights, num_classes):
    """Helper function to build and modify an EfficientNet model."""
    if model_name == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(weights=weights)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {model_name}")

    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model


# --- Model and Weight Registries ---

WEIGHTS_MAPPING = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
    "vit_l_32": models.ViT_L_32_Weights.DEFAULT,
    "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.DEFAULT,
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
}

MODEL_REGISTRY = {
    "resnet18": _build_resnet,
    "resnet34": _build_resnet,
    "vit_b_16": _build_vit,
    "vit_l_32": _build_vit,
    "efficientnet_v2_l": _build_efficientnet,
    "efficientnet_v2_s": _build_efficientnet,
}


def get_model(model_name, pretrained=True, num_classes=3):
    """
    Loads a model from the registry, replaces the classification head, and loads pre-trained weights if specified.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_REGISTRY.keys())}")

    builder = MODEL_REGISTRY[model_name]
    weights = WEIGHTS_MAPPING[model_name] if pretrained else None
    model = builder(model_name=model_name, weights=weights, num_classes=num_classes)
    return model
