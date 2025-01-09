import torch
import torch.nn as nn


class Bhanu(nn.Module):
    """
    A Convolutional Neural Network (CNN) architecture for binary classification of grayscale images.

    Args:
        cfg (Config): Configuration object containing the model hyperparameters.

    Input:
        x: Grayscale images of shape (batch_size, 1, height, width)

    Returns:
        torch.Tensor: Binary classification logits of shape (batch_size, 1)
    """

    def __init__(self, cfg):
        # Initialize the PyTorch ViT implementation
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
