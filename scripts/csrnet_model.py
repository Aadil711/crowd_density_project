"""
CSRNet model architecture for crowd counting.
Based on: https://arxiv.org/abs/1802.10062
"""
import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        # Frontend: VGG-16 first 10 conv layers
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:23])  # Up to conv4_3
        
        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Output layer: density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Initialize backend weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def load_model(weights_path=None):
    """Load CSRNet model with optional pretrained weights."""
    model = CSRNet()
    
    if weights_path:
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print("Using model with VGG-16 pretrained frontend only")
    
    model.eval()
    return model
