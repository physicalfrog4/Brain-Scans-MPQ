import torch
from torchvision.models import vgg19
from torchvision import ops
from ultralytics import YOLO

from PIL import Image

#not used
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

#not used
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

#Currently used
#A wrapper class to overwrite to string functions so that model isn't printed on server side
class YoloModel(YOLO):
    def __init__(self, *args, **kwargs):
        super(YoloModel, self).__init__(*args, **kwargs)
    def __str__(self):
        return ""
    def __repr__(self):
        return ""

#Currently Using
class roiVGGYolo(torch.nn.Module):
    def __init__(self, numROIs: int, tsfms):
        super(roiVGGYolo, self).__init__()
        #Make VGG Instance for feature extraction
        self.vgg = vgg19(weights = "DEFAULT")

        #Get layers to use for feature extraction and freeze the model params so they aren't updated during training
        self.vggConvFeatures = self.vgg.features[:35]
        for params in self.vgg.parameters():
            params.requires_grad = False

        #Make yolo instance and freeze parameters so they aren't updated during training
        self.yolo = YoloModel("yolov8n.pt")
        for params in self.yolo.parameters():
            params.requires_grad = False

        #Create the MLP which is just a series of linear layers with relu
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, numROIs),
        )

        #Save the torch transforms for the YOLO model
        self.tsfms = tsfms

    def forward(self, img, imgPaths):
        #extract vgg features from image
        convFeatures = self.vggConvFeatures(img)

        #transform the original images such as it's compatible with the YOLO model
        yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]

        #Make YOLO predictions on images and get bounding box data
        yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
        boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
        finalFMRIs = []
        indices = [] #saved indexes to use that have bounding box info from yolo
        count = 0

        #for each detected bounding box
        for boundingBoxData in boundingBoxDataAllImages:
            #if no bounding boxes, continue (ignore image)
            if len(boundingBoxData) == 0:
                continue

            #pool the features in the regions that overlap with a bounding box. returns a result for each detected object
            objectROIPools = ops.roi_pool(convFeatures, [boundingBoxData], output_size = (7,7)) #output shape (num objects, 512, 7, 7)
            fmriPieces = []

            #for the pooled results for each object, predict the partial fmri data
            for objectROIPool in objectROIPools:
                input = torch.flatten(objectROIPool) #flatten data to pass into mlp, shape (1, 25088)
                fmriPieces.append(self.MLP(input)) #save partial fmri prediction

            #sum over all partial fmri data
            totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
            finalFMRIs.append(totalFMRI)
            indices.append(count)
            count+=1
        return torch.stack(finalFMRIs), indices 

    #extract bounding box data for each of the yolo results objects
    def getMappedBoundingBox(self, yoloResults):
        mappedBoxes = []
        for result in yoloResults:
            #get normalized top left and bottom right coordinates for the bounding box
            boundingBoxData = result.xyxyn

            #Get interested cells in 7x7 matrix for ROI pooling                      topLeftX               topLeftY               bottomRightX           #bottomRightY
            boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY = boundingBoxData[:, 0], boundingBoxData[:, 1], boundingBoxData[:, 2], boundingBoxData[:, 3]

            #Transform normalized range (0 to 1) to (0 to 7)
            transformedBoundingBoxStartX, transformedBoundingBoxStartY, transformedBoundingBoxEndX, transformedBoundingBoxEndY = boundingBoxStartX * 7, boundingBoxStartY * 7, boundingBoxEndX * 7, boundingBoxEndY * 7
            startCellX = torch.floor(transformedBoundingBoxStartX)
            startCellY = torch.floor(transformedBoundingBoxStartY)
            endCellX = torch.ceil(transformedBoundingBoxEndX)
            endCellY = torch.ceil(transformedBoundingBoxEndY)
            mappedBoxes.append(torch.hstack((startCellX.reshape(-1,1), startCellY.reshape(-1,1), endCellX.reshape(-1,1), endCellY.reshape(-1,1))))
        return mappedBoxes

    #Overwrite functions so that the model isn't printed on server side after each call to train() or eval()
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

#Currently Using
class roiVGGYoloWithGradCam(torch.nn.Module):
    def __init__(self, numROIs: int, tsfms):
        super(roiVGGYoloWithGradCam, self).__init__()
        #Make VGG Instance for feature extraction
        self.vgg = vgg19(weights = "DEFAULT")

        #Get layers to use for feature extraction 
        self.vggConvFeatures = self.vgg.features[:35]

        #Make yolo instance
        self.yolo = YoloModel("yolov8n.pt")

        #Create the MLP which is just a series of linear layers with relu
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, numROIs),
        )

        #Save the torch transforms for the YOLO model
        self.tsfms = tsfms
        #variable for the gradients used in grad cam
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, img, imgPaths):
        #extract vgg features from image
        convFeatures = self.vggConvFeatures(img)

        #define function to call during back prop
        hook = convFeatures.register_hook(self.activations_hook)

        #transform the original images such as it's compatible with the YOLO model
        yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]

        #Make YOLO predictions on images and get bounding box data
        yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
        boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
        finalFMRIs = []
        indices = [] #saved indexes to use that have bounding box info from yolo
        count = 0

        #for each detected bounding box
        for boundingBoxData in boundingBoxDataAllImages:
            #if no bounding boxes, continue
            if len(boundingBoxData) == 0:
                continue

            #pool the features in the regions that overlap with a bounding box. returns a result for each detected object
            objectROIPools = ops.roi_pool(convFeatures, [boundingBoxData], output_size = (7,7)) #output shape (num objects, 512, 7, 7)
            fmriPieces = []

            #for the pooled results for each object, predict the partial fmri data
            for objectROIPool in objectROIPools:
                input = torch.flatten(objectROIPool)  #flatten data to pass into mlp, shape (1, 25088)
                fmriPieces.append(self.MLP(input)) #save partial fmri prediction

            #sum over all partial fmri data
            totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
            finalFMRIs.append(totalFMRI)
            indices.append(count)
            count+=1
        return torch.stack(finalFMRIs), indices 

    #extract bounding box data for each of the yolo results objects
    def getMappedBoundingBox(self, yoloResults):
        mappedBoxes = []
        for result in yoloResults:
            #get normalized top left and bottom right coordinates for the bounding box
            boundingBoxData = result.xyxyn

            #Get interested cells in 7x7 matrix for ROI pooling                      topLeftX               topLeftY               bottomRightX           #bottomRightY
            boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY = boundingBoxData[:, 0], boundingBoxData[:, 1], boundingBoxData[:, 2], boundingBoxData[:, 3]

            #Transform normalized range (0 to 1) to (0 to 7)
            transformedBoundingBoxStartX, transformedBoundingBoxStartY, transformedBoundingBoxEndX, transformedBoundingBoxEndY = boundingBoxStartX * 7, boundingBoxStartY * 7, boundingBoxEndX * 7, boundingBoxEndY * 7
            startCellX = torch.floor(transformedBoundingBoxStartX)
            startCellY = torch.floor(transformedBoundingBoxStartY)
            endCellX = torch.ceil(transformedBoundingBoxEndX)
            endCellY = torch.ceil(transformedBoundingBoxEndY)
            mappedBoxes.append(torch.hstack((startCellX.reshape(-1,1), startCellY.reshape(-1,1), endCellX.reshape(-1,1), endCellY.reshape(-1,1))))
        return mappedBoxes

    def get_activation_gradient(self):
        return self.gradients

    def get_activations(self, img):
        return self.vggConvFeatures(img)

    #Overwrite functions so that the model isn't printed on server side after each call to train() or eval()
    def __str__(self):
        return ""
    def __repr__(self):
        return ""
