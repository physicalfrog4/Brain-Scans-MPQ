import os
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import ops
from PIL import Image
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from torch.optim import lr_scheduler



class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = subj#format(subj, '02')
        self.data_dir = data_dir#os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        # self.subject_submission_dir = os.path.join(self.parent_submission_dir, 'subj' + self.subj)

def normalize_fmri_data(data):
    clip_percentile = 0.05
    # Clip extreme values to handle outliers
    min_clip = np.percentile(data, clip_percentile)
    max_clip = np.percentile(data, 100 - clip_percentile)
    clipped_data = np.clip(data, min_clip, max_clip)
    # Scale the clipped data to the range [0, 1]
    min_value = np.min(clipped_data)
    max_value = np.max(clipped_data)
    print(min_value, max_value)
    if max_value == min_value:
        normalized_data = np.zeros_like(clipped_data)
    else:
        normalized_data = (clipped_data - min_value) / (max_value - min_value)
    return normalized_data, min_value, max_value

#Calculates the average fmri value for each ROI that has data
def getAvgROI(parentFolderDir: str, subj: int, fmriData, hemi: str = "l"):
    rois = np.array(["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"])
    avgRoiValues = np.zeros((len(fmriData), len(rois)))
    for i in range(len(rois)):
        roi = rois[i]
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'
        roiMap = np.load(f"{parentFolderDir}/subj0{subj}/roi_masks/mapping_{roi_class}.npy", allow_pickle=True).item()
        challenge_roi_class = np.load(f"{parentFolderDir}/subj0{subj}/roi_masks/{hemi}h.{roi_class}_challenge_space.npy")
        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roiMap.keys())[list(roiMap.values()).index(roi)]
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
        # print(roi_mapping)       
        vals = fmriData[:,np.where(challenge_roi)[0]].mean(axis = 1)
        avgRoiValues[:, i] = vals
    mask = np.arange(len(avgRoiValues[0]))
    print(mask)
    mask = mask[~np.isnan(avgRoiValues.max(axis=0))]
    print(mask)
    return rois[mask], avgRoiValues[:, mask]

#currently using
#Creates dataset with all training images for a specific subject 
class AlgonautsDataset(Dataset):
    def __init__(self, parentDir: str, subj: int, dataIdxs: list = None, transform = None):
        self.imagesPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_images/")
        self.fmriPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_fmri/")
        self.imagePaths = np.array(os.listdir(self.imagesPath))
        self.lhFMRI = np.load(os.path.join(self.fmriPath, "lh_training_fmri.npy"))
        self.rhFMRI = np.load(os.path.join(self.fmriPath, "rh_training_fmri.npy"))
        self.normalLhFMRI, self.lh_data_min, self.lh_data_max = normalize_fmri_data(self.lhFMRI)
        self.normalRhFMRI, self.rh_data_min, self.rh_data_max = normalize_fmri_data(self.rhFMRI)
        # self.lhROIs, self.lhAvgFMRI = getAvgROI(parentDir, subj, self.lhFMRI)
        # self.rhROIs, self.rhAvgFMRI = getAvgROI(parentDir, subj, self.rhFMRI, hemi="r")
        self.transform = transform
        if dataIdxs is not None:
            self.imagePaths = self.imagePaths[dataIdxs]
            self.lhFMRI = self.lhFMRI[dataIdxs]
            self.rhFMRI = self.rhFMRI[dataIdxs]
            self.normalLhFMRI = self.normalLhFMRI[dataIdxs]
            self.normalRhFMRI = self.normalRhFMRI[dataIdxs]
    def __len__(self):
        return len(self.imagePaths)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = os.path.join(self.imagesPath, self.imagePaths[idx])
        image = Image.open(imagePath)
        if self.transform:
            image = self.transform(image)
        lh, rh = self.lhFMRI[idx], self.rhFMRI[idx]
        normalLh, normalRh = self.normalLhFMRI[idx], self.normalRhFMRI[idx]
        return image, imagePath, torch.tensor(lh, dtype=torch.float32), torch.tensor(rh, dtype=torch.float32), torch.tensor(normalLh, dtype=torch.float32), torch.tensor(normalRh, dtype=torch.float32)

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
        # self.yolo = YoloModel("yolov8n.pt")
        # for params in self.yolo.parameters():
        #     params.requires_grad = False
        #Create the MLP which is just a series of linear layers with relu
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(100352, 25088),
            torch.nn.ReLU(),
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, numROIs),
        )
        #Save the torch transforms for the YOLO model
        self.tsfms = tsfms
    def forward(self, img):
        #extract vgg features from image
        print(img.shape)
        convFeatures = self.vggConvFeatures(img)
        print(convFeatures.shape)
        convFeatures = torch.flatten(convFeatures,1)
        print(convFeatures.shape)
        return self.MLP(convFeatures)
        # #transform the original images such as it's compatible with the YOLO model
        # yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]
        # #Make YOLO predictions on images and get bounding box data
        # yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
        # boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
        # finalFMRIs = []
        # indices = [] #saved indexes to use that have bounding box info from yolo
        # count = 0
        # #for each detected bounding box
        # for boundingBoxData in boundingBoxDataAllImages:
        #     #if no bounding boxes, continue (ignore image)
        #     if len(boundingBoxData) == 0:
        #         count+=1
        #         continue
        #     #pool the features in the regions that overlap with a bounding box. returns a result for each detected object
        #     objectROIPools = ops.roi_pool(convFeatures, [boundingBoxData], output_size = (7,7)) #output shape (num objects, 512, 7, 7)
        #     fmriPieces = []
        #     #for the pooled results for each object, predict the partial fmri data
        #     for objectROIPool in objectROIPools:
        #         input = torch.flatten(objectROIPool) #flatten data to pass into mlp, shape (1, 25088)
        #         fmriPieces.append(self.MLP(input)) #save partial fmri prediction
        #     #sum over all partial fmri data
        #     totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
        #     finalFMRIs.append(totalFMRI)
        #     indices.append(count)
        #     count+=1
        # return torch.stack(finalFMRIs), indices 
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

platform = 'jupyter_notebook'
device = 'cuda:0'
device = torch.device(device)
# setting up the directories and ARGS
data_dir = '/home/vislab-001/Documents/algonauts_2023_challenge_data/'#../MQP/algonauts_2023_challenge_data/'
parent_submission_dir = '../submission'
subj = 1 # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
# args

args = argObj(data_dir, parent_submission_dir, subj)
words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
            'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing', 
             'computer', 'fruit', 'vegetable', 'tool']

#define transforms for the images to be passed into the vgg model
tsfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

#define transforms for the images to be passed into the YOLO model
yoloTsfms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

trainingDataset = AlgonautsDataset(args.data_dir, args.subj, transform=tsfms)
trainIdxs, validIdxs = train_test_split(range(len(trainingDataset.imagePaths)), train_size=0.9, random_state = 42)
lhNumROIs = len(trainingDataset.lhFMRI[0])
rhNumROIs = len(trainingDataset.rhFMRI[0])
trainSubset = Subset(trainingDataset, trainIdxs)
trainDataLoader = DataLoader(trainSubset, batch_size = 32, shuffle = True)
validSubset = Subset(trainingDataset, validIdxs)
validDataLoader = DataLoader(validSubset, batch_size = 32, shuffle = False)#change

vgg19Model = vgg19(weights = "DEFAULT")
featureExtractor = vgg19Model.features[:35]

mp7 = torch.nn.MaxPool2d(7, padding=0, stride=1)
mp5 = torch.nn.MaxPool2d(5, padding=0, stride=1)
mp3 = torch.nn.MaxPool2d(3, padding=0, stride=1)
mp1 = torch.nn.MaxPool2d(1, padding=0, stride=1)