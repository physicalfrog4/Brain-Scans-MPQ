import torch
from torchvision.models import vgg19
from torchvision import ops
from ultralytics import YOLO

from PIL import Image

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

class roiVGG(torch.nn.Module):
    def __init__(self, numClasses: int, cocoVGGWeights, numROIs: int, device):
        super(roiVGG, self).__init__()
        self.cocoVgg19 = CocoVGG(numClasses)
        self.cocoVgg19.load_state_dict(torch.load(cocoVGGWeights, map_location=device))
        self.convFeatures = self.cocoVgg19.features[:36]
        self.maxPool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
        self.adaptiveAvgPool = torch.nn.AdaptiveAvgPool2d(output_size = (7, 7))
        self.cocoClassifier = self.cocoVgg19.classifier
        self.roiClassifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 25088, out_features = 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 4096, out_features = 1024),#1024
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 1024, out_features = numROIs)
        )
        self.gradients = None
    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, img):
        extractedFeatures = self.convFeatures(img)
        # hook = extractedFeatures.register_hook(self.activations_hook)
        intermediateOutput = self.maxPool(extractedFeatures)
        intermediateOutput = self.adaptiveAvgPool(intermediateOutput)
        print(intermediateOutput.shape)
        intermediateOutput = torch.flatten(intermediateOutput, 1)
        return self.cocoClassifier(intermediateOutput), self.roiClassifier(intermediateOutput)
    def get_activation_gradient(self):
        return self.gradients
    def get_activations(self, img):
        return self.convFeatures(img)
    def __str__(self):
        return ""
    def __repr__(self):
        return ""

class YoloModel(YOLO):
    def __init__(self, *args, **kwargs):
        super(YoloModel, self).__init__(*args, **kwargs)
    def __str__(self):
        return ""
    def __repr__(self):
        return ""

class roiVGGYolo(torch.nn.Module):
    def __init__(self, numROIs: int, tsfms):
        super(roiVGGYolo, self).__init__()
        self.vgg = vgg19(weights = "DEFAULT")
        self.vggConvFeatures = self.vgg.features[:35]
        for params in self.vgg.parameters():
            params.requires_grad = False
        self.yolo = YoloModel("yolov8n.pt")
        for params in self.yolo.parameters():
            params.requires_grad = False
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, numROIs),
        )
        self.tsfms = tsfms
    def forward(self, img, imgPaths):
        # pooling = self.vgg.features(img)
        # pooling = self.vgg.avgpool(pooling)
        convFeatures = self.vggConvFeatures(img)
        yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]
        yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
        boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
        finalFMRIs = []
        indices = []
        count = 0
        for boundingBoxData in boundingBoxDataAllImages:
            # print(boundingBoxData)
            # print(imgPaths[count])
            if len(boundingBoxData) == 0:
                continue
            objectROIPools = ops.roi_pool(convFeatures, [boundingBoxData], output_size = (7,7))
            fmriPieces = []
            for objectROIPool in objectROIPools:
                # print("objROIP")
                # print(objectROIPool.shape)
                # print(objectROIPool)
                input = torch.flatten(objectROIPool)
                fmriPieces.append(self.MLP(input))
            totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
            finalFMRIs.append(totalFMRI)
            indices.append(count)
            count+=1
            # print("fmriPieces")
            # print(f"Shape: ({len(fmriPieces)}, {fmriPieces[0].shape})")
            # print(fmriPieces)
        return torch.stack(finalFMRIs), indices 
    def getMappedBoundingBox(self, yoloResults):
        mappedBoxes = []
        for result in yoloResults:
            boundingBoxData = result.xyxyn
            boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY = boundingBoxData[:, 0], boundingBoxData[:, 1], boundingBoxData[:, 2], boundingBoxData[:, 3]
            transformedBoundingBoxStartX, transformedBoundingBoxStartY, transformedBoundingBoxEndX, transformedBoundingBoxEndY = boundingBoxStartX * 7, boundingBoxStartY * 7, boundingBoxEndX * 7, boundingBoxEndY * 7
            startCellX = torch.floor(transformedBoundingBoxStartX)
            startCellY = torch.floor(transformedBoundingBoxStartY)
            endCellX = torch.ceil(transformedBoundingBoxEndX)
            endCellY = torch.ceil(transformedBoundingBoxEndY)
            mappedBoxes.append(torch.hstack((startCellX.reshape(-1,1), startCellY.reshape(-1,1), endCellX.reshape(-1,1), endCellY.reshape(-1,1))))
        # print("mapped boxes")
        # print(f"Shape: ({len(mappedBoxes)}, {mappedBoxes[0].shape})")
        # print(mappedBoxes)
        return mappedBoxes
    def __str__(self):
        return ""
    def __repr__(self):
        return ""
