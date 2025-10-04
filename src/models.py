import torch.nn as nn
from torchvision import models


def get_model(model_name="resnet18", pretrained=True, num_classes=1):
    """
    Loads a pre-trained model and replaces the final classification layer.

    Args:
        model_name (str): The name of the model architecture to load (e.g., "resnet18").
        pretrained (bool): Whether to load pre-trained ImageNet weights.
        num_classes (int): The number of output features for the final layer.

    Returns:
        torch.nn.Module: The modified model.
    """
    # Load the pre-trained model
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    # Add other models here as needed
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Get the number of input features to the final fully connected layer
    num_ftrs = model.fc.in_features

    # Replace the final layer with a new one for our specific task
    # The output is a single logit for binary classification.
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
