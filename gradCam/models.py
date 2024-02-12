import torch
from torchvision.models import vgg19

class CocoVGG (torch.nn.Module):
    def __init__(self, numClasses):
        super(CocoVGG, self).__init__()
        self.vgg = vgg19(weights="DEFAULT")
        self.features = self.vgg.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 25088, out_features = 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 4096, out_features = 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 1024, out_features = numClasses),
            torch.nn.Softmax(dim=1)
        )
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)
    def forward(self, img):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def __str__(self):
        return ""
    def __repr__(self):
        return ""
