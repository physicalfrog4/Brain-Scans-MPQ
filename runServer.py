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
        self.yolo = YoloModel("yolov8n.pt")
        for params in self.yolo.parameters():
            params.requires_grad = False
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
    def forward(self, img, imgPaths):
        #extract vgg features from image
        convFeatures = self.vggConvFeatures(img)
        convFeatures = torch.flatten(convFeatures,1)
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
device = 'cuda:1'
device = torch.device(device)
# setting up the directories and ARGS
data_dir = './algonauts_2023_challenge_data/'#../MQP/algonauts_2023_challenge_data/'
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
trainDataLoader = DataLoader(trainSubset, batch_size = 128, shuffle = True)
validSubset = Subset(trainingDataset, validIdxs)
validDataLoader = DataLoader(validSubset, batch_size = 128, shuffle = False)#change
#Create VGG and YOLO model
lhModel = roiVGGYolo(lhNumROIs, yoloTsfms).to(device)
# rhModel = roiVGGYolo(rhNumROIs, yoloTsfms).to(device)
#Define optimzer params and loss function
learningRate = 0.001
lhOptim = torch.optim.Adam(lhModel.parameters(), learningRate)#,  weight_decay=1e-4
lhScheduler = lr_scheduler.StepLR(lhOptim, step_size=10, gamma=0.1)
# rhOptim = torch.optim.Adam(rhModel.parameters(), learningRate)#,  weight_decay=1e-4
# rhScheduler = lr_scheduler.StepLR(rhOptim, step_size=10, gamma=0.1)
criterion = torch.nn.MSELoss()
#Training for specified epochs
epochs = 25

# lhPredictions = []
# lhActual = []
# allLhIndices = np.array([])
# rhPredictions = []
# allRhIndices = np.array([])
# first = True
# with torch.no_grad():
#     for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
#         #Get data in batch
#         img, imgPaths, _, _, normalLhFMRI, _ = data
#         img = img.to(device)
#         # normalLhFMRI = normalLhFMRI.to(device)
#         # normalRhFMRI = normalRhFMRI.to(device)
#         #make predictions
#         lhPred, lhIndices = lhModel(img, imgPaths)
#         normalLhFMRI = normalLhFMRI[lhIndices]
#         # rhPred, rhIndices = rhModel(img, imgPaths)
#         # print(lhIndices)
#         #
#         if first:
#             first = False
#             lhPredictions.extend(lhPred.cpu().numpy())
#             lhActual.extend(normalLhFMRI.cpu().numpy())
#             allLhIndices = np.concatenate((allLhIndices, np.array(lhIndices)))
#             # allLhIndices.extend(lhIndices )
#             # rhPredictions.extend(rhPred.cpu().numpy())
#             # allRhIndices = np.concatenate((allRhIndices, np.array(rhIndices)))
#             # allRhIndices.extend(rhIndices)
#         else:
#             lhPredictions.extend(lhPred.cpu().numpy())
#             lhActual.extend(normalLhFMRI.cpu().numpy())
#             allLhIndices = np.concatenate((allLhIndices, np.array(lhIndices) + allLhIndices[-1] + 1))
#             # allLhIndices.extend(lhIndices )
#             # rhPredictions.extend(rhPred.cpu().numpy())
#             # allRhIndices = np.concatenate((allRhIndices, np.array(rhIndices) + allRhIndices[-1] + 1))
#             # allRhIndices.extend(rhIndices)
    



# lhPredictions = np.array(lhPredictions)
# lhActual = np.array(lhActual)
# rhPredictions = np.array(rhPredictions)


def pearsonR(output, target):
    vx = output - torch.mean(output)
    vy = target - torch.mean(target)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -cost




def combinedLoss (pred, actual, mseWeight):
    MSE = torch.nn.MSELoss()
    mseLoss = MSE(pred, actual)
    pearsonLoss = pearsonR(pred, actual)
    return mseWeight * mseLoss + (1 - mseWeight) * pearsonLoss

mseWeight = 0.8

for epoch in range(epochs):
    #Variables to hold information on training progress
    print(f"Epoch {epoch}")
    avgLhTrainingLoss = 0
    avgLhEvalLoss = 0
    avgLhTrainingR2Score = 0
    avgLhEvalR2Score = 0
    avgLhTrainingMSE = 0
    avgLhEvalMSE = 0
    # avgRhTrainingLoss = 0
    # avgRhEvalLoss = 0
    # avgRhTrainingR2Score = 0
    # avgRhEvalR2Score = 0
    for data in tqdm(trainDataLoader, desc="Training", unit="batch"): 
        #Get data in batch
        img, imgPaths, lhFMRI, _, _, _ = data
        img = img.to(device)
        lhFMRI = lhFMRI.to(device)
        # normalRhFMRI = normalRhFMRI.to(device)
        lhOptim.zero_grad()
        # rhOptim.zero_grad()
        #make predictions
        lhPred = lhModel(img, imgPaths)
        # rhPred, rhIndices = rhModel(img, imgPaths)
        #only use data for images that had bounding box data
        # normalRhFMRI = normalRhFMRI[rhIndices]
        #evaluate based on loss function
        lhLoss = combinedLoss(lhPred, lhFMRI, mseWeight)
        lhLoss.backward()
        lhOptim.step()
        # rhLoss = criterion(rhPred, normalRhFMRI)
        # rhLoss.backward()
        # rhOptim.step()    
        #sum r2 scores and loss values for batch
        avgLhTrainingLoss += lhLoss.item()
        avgLhTrainingR2Score += r2_score(lhPred.detach().cpu().numpy(), lhFMRI.detach().cpu().numpy())
        avgLhTrainingMSE += criterion(lhPred, lhFMRI).item()
        # avgRhTrainingLoss += rhLoss.item()
        # avgRhTrainingR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
    with torch.no_grad():
        for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
            #Get data in batch
            img, imgPaths, lhFMRI, _, _, _ = data
            img = img.to(device)
            lhFMRI = lhFMRI.to(device)
            # normalRhFMRI = normalRhFMRI.to(device)
            #make predictions
            lhPred = lhModel(img, imgPaths)
            # rhPred, rhIndices = rhModel(img, imgPaths)
            #only use data for images that had bounding box data
            # normalRhFMRI = normalRhFMRI[rhIndices]
            #evaluate based on loss function
            lhEvalLoss = combinedLoss(lhPred, lhFMRI, mseWeight)
            # rhEvalLoss = criterion(rhPred, normalRhFMRI)
            #sum r2 scores and loss values for batch
            avgLhEvalLoss += lhEvalLoss.item()
            avgLhEvalR2Score += r2_score(lhPred.detach().cpu().numpy(), lhFMRI.detach().cpu().numpy())
            avgLhEvalMSE += criterion(lhPred, lhFMRI).item()
            # avgRhEvalLoss += rhEvalLoss.item()
            # avgRhEvalR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
    lhScheduler.step()
    # rhScheduler.step()
    #calculate metrics for epoch
    validLhMse = avgLhEvalLoss / len(validDataLoader)
    validLhR2 = avgLhEvalR2Score / len(validDataLoader)
    # validRhMse = avgRhEvalLoss / len(validDataLoader)
    # validRhR2 = avgRhEvalR2Score / len(validDataLoader)
    print(f"lh using lr = {lhScheduler.get_last_lr()} TrainingLoss: {avgLhTrainingLoss / len(trainDataLoader)}, ValidLoss: {validLhMse}, trainR2 = {avgLhTrainingR2Score / len(trainDataLoader)}, evalR2= {validLhR2}, trainMSE = {avgLhTrainingMSE / len(trainDataLoader)}, validMSE = {avgLhEvalMSE / len(validDataLoader)}")
    # print(f"rh using lr = {rhScheduler.get_last_lr()} TrainingMSE: {avgRhTrainingLoss / len(trainDataLoader)}, ValidMSE: {validRhMse}, trainR2 = {avgRhTrainingR2Score / len(trainDataLoader)}, evalR2= {validRhR2}")

# lh using lr = [1.0000000000000002e-06] TrainingMSE: 0.2635801968074614, ValidMSE: 0.2618712689727545, trainR2 = -424.4069813331531, evalR2= -499.2288785518732
# rh using lr = [1.0000000000000002e-06] TrainingMSE: 0.2691644757024704, ValidMSE: 0.2671606373041868, trainR2 = -421.10467548117356, evalR2= -495.50196343046673

# bestLhModel = {
#     "fold": 0,
#     "mse": float('inf'),  # Set to positive infinity initially
#     "r2": float('-inf'),  # Set to negative infinity initially
#     "params": None  # Initialize with None, will be updated with model state dict
# }

# bestRhModel = {
#     "fold": 0,
#     "mse": float('inf'),  # Set to positive infinity initially
#     "r2": float('-inf'),  # Set to negative infinity initially
#     "params": None  # Initialize with None, will be updated with model state dict
# }

# #Define KFold technique
k_folds = 5
skf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
skf.split
# # Loop over folds
# for fold, (train_idxs, val_idxs) in enumerate(skf.split(trainingDataset.imagePaths)):
#     #Create subset of dataset with the corresponding training and validation indexes for this fold
#     trainSubset = Subset(trainingDataset, train_idxs)
#     trainDataLoader = DataLoader(trainSubset, batch_size = 128, shuffle = True)
#     validSubset = Subset(trainingDataset, val_idxs)
#     validDataLoader = DataLoader(validSubset, batch_size = 128, shuffle = True)
#     #Create VGG and YOLO model
#     lhModel = roiVGGYolo(lhNumROIs, yoloTsfms).to(device)
#     rhModel = roiVGGYolo(rhNumROIs, yoloTsfms).to(device)
#     #Define optimzer params and loss function
#     learningRate = 0.00001
#     lhOptim = torch.optim.Adam(lhModel.parameters(), learningRate)#,  weight_decay=1e-4
#     lhScheduler = lr_scheduler.StepLR(lhOptim, step_size=10, gamma=0.1)
#     rhOptim = torch.optim.Adam(rhModel.parameters(), learningRate)#,  weight_decay=1e-4
#     rhScheduler = lr_scheduler.StepLR(rhOptim, step_size=10, gamma=0.1)
#     criterion = torch.nn.MSELoss()
#     #Training for specified epochs
#     epochs = 15
#     for epoch in range(epochs):
#         #Variables to hold information on training progress
#         print(f"Epoch {epoch}")
#         avgLhTrainingLoss = 0
#         avgLhEvalLoss = 0
#         avgLhTrainingR2Score = 0
#         avgLhEvalR2Score = 0
#         avgRhTrainingLoss = 0
#         avgRhEvalLoss = 0
#         avgRhTrainingR2Score = 0
#         avgRhEvalR2Score = 0
#         for data in tqdm(trainDataLoader, desc="Training", unit="batch"): 
#             #Get data in batch
#             img, imgPaths, _, _, normalLhFMRI, normalRhFMRI = data
#             img = img.to(device)
#             normalLhFMRI = normalLhFMRI.to(device)
#             normalRhFMRI = normalRhFMRI.to(device)
#             lhOptim.zero_grad()
#             rhOptim.zero_grad()
#             #make predictions
#             lhPred, lhIndices = lhModel(img, imgPaths)
#             rhPred, rhIndices = rhModel(img, imgPaths)
#             #only use data for images that had bounding box data
#             normalLhFMRI = normalLhFMRI[lhIndices]
#             normalRhFMRI = normalRhFMRI[rhIndices]
#             #evaluate based on loss function
#             lhLoss = criterion(lhPred, normalLhFMRI)
#             lhLoss.backward()
#             lhOptim.step()
#             rhLoss = criterion(rhPred, normalRhFMRI)
#             rhLoss.backward()
#             rhOptim.step()    
#             #sum r2 scores and loss values for batch
#             avgLhTrainingLoss += lhLoss.item()
#             avgLhTrainingR2Score += r2_score(lhPred.detach().cpu().numpy(), normalLhFMRI.detach().cpu().numpy())
#             avgRhTrainingLoss += rhLoss.item()
#             avgRhTrainingR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
#         with torch.no_grad():
#             for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
#                 #Get data in batch
#                 img, imgPaths, _, _, normalLhFMRI, normalRhFMRI = data
#                 img = img.to(device)
#                 normalLhFMRI = normalLhFMRI.to(device)
#                 normalRhFMRI = normalRhFMRI.to(device)
#                 #make predictions
#                 lhPred, lhIndices = lhModel(img, imgPaths)
#                 rhPred, rhIndices = rhModel(img, imgPaths)
#                 #only use data for images that had bounding box data
#                 normalLhFMRI = normalLhFMRI[lhIndices]
#                 normalRhFMRI = normalRhFMRI[rhIndices]
#                 #evaluate based on loss function
#                 lhEvalLoss = criterion(lhPred, normalLhFMRI)
#                 rhEvalLoss = criterion(rhPred, normalRhFMRI)
#                 #sum r2 scores and loss values for batch
#                 avgLhEvalLoss += lhEvalLoss.item()
#                 avgLhEvalR2Score += r2_score(lhPred.detach().cpu().numpy(), normalLhFMRI.detach().cpu().numpy())
#                 avgRhEvalLoss += rhEvalLoss.item()
#                 avgRhEvalR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
#         lhScheduler.step()
#         rhScheduler.step()
#         #calculate metrics for epoch
#         validLhMse = avgLhEvalLoss / len(validDataLoader)
#         validLhR2 = avgLhEvalR2Score / len(validDataLoader)
#         validRhMse = avgRhEvalLoss / len(validDataLoader)
#         validRhR2 = avgRhEvalR2Score / len(validDataLoader)
#         print(f"lh using lr = {lhScheduler.get_last_lr()} TrainingMSE: {avgLhTrainingLoss / len(trainDataLoader)}, ValidMSE: {validLhMse}, trainR2 = {avgLhTrainingR2Score / len(trainDataLoader)}, evalR2= {validLhR2}")
#         print(f"rh using lr = {rhScheduler.get_last_lr()} TrainingMSE: {avgRhTrainingLoss / len(trainDataLoader)}, ValidMSE: {validRhMse}, trainR2 = {avgRhTrainingR2Score / len(trainDataLoader)}, evalR2= {validRhR2}")
#         #save model params and info if better than recorded validation mse
#         if validLhMse < bestLhModel["mse"]:
#             print("BESTMODEL SO FAR")
#             bestLhModel["fold"] = fold
#             bestLhModel["mse"] = validLhMse
#             bestLhModel["r2"] = validLhR2
#             bestLhModel["params"] = lhModel.state_dict()  # Update with the current model's state dict
#         if validRhMse < bestRhModel["mse"]:
#             print("BESTMODEL SO FAR")
#             bestRhModel["fold"] = fold
#             bestRhModel["mse"] = validRhMse
#             bestRhModel["r2"] = validRhR2
#             bestRhModel["params"] = rhModel.state_dict()  # Update with the current model's state dict

# #save best model params and data
# torch.save(bestLhModel["params"], './5FoldLhAllFMRIBestModel.pth')
# bestModelData = np.array([bestLhModel["fold"], bestLhModel["mse"], bestLhModel["r2"]])
# np.save("5FoldLhAllFMRIBestModelData.npy", bestModelData)

# torch.save(bestRhModel["params"], './5FoldRhAllFMRIBestModel.pth')
# bestModelData = np.array([bestRhModel["fold"], bestRhModel["mse"], bestRhModel["r2"]])
# np.save("5FoldRhAllFMRIBestModelData.npy", bestModelData)
