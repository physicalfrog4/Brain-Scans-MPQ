import os
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import ops

from ultralytics import YOLO, settings
settings.update({
    "clearml" : False,          
    "comet" : False,           
    "dvc" : False,             
    "hub" : False,     
    "mlflow" : False,          
    "neptune" : False,         
    "raytune" : False,         
    "tensorboard" : False,     
    "wandb" : False     
})

import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


def getClassImages(dataDir:str, imgDataFolerDir: str, subj: int, className: str):
    pickleFilePath = os.path.join(imgDataFolerDir, f"subj0{subj}ImgData.pkl")
    imgData = pd.read_pickle(pickleFilePath)
    imageIDsInClass = imgData[imgData["classLabel"] == "person"]["nsdId"].values
    trainingImagesPath = os.path.join(dataDir, f"subj0{1}/training_split/training_images/")
    subjImages = os.listdir(trainingImagesPath)
    imagesInClass = []
    trainingIDsInClass = []
    for image in subjImages:
        imageSplits = image.split("-") # ['train', '0004_nsd', '00085.png']
        trainingID = int(imageSplits[1].rstrip("_nsd")) - 1
        nsdID = imageSplits[-1] # removes 'train-0001_nsd-' and keeps '00013.png'
        nsdID = nsdID.rstrip(".png") #removes png file type
        nsdID = int(nsdID)
        if nsdID in imageIDsInClass:
            imagesInClass.append(os.path.join(trainingImagesPath, image))
            trainingIDsInClass.append(trainingID)
    return np.array(imagesInClass), np.array(trainingIDsInClass)

    
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


class AlgonautsDataset(Dataset):
    def __init__(self, parentDir: str, subj: int, dataIdxs: list = None, transform = None):
        self.imagesPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_images/")
        self.fmriPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_fmri/")
        self.imagePaths = np.array(os.listdir(self.imagesPath))
        self.lhFMRI = np.load(os.path.join(self.fmriPath, "lh_training_fmri.npy"))
        self.rhFMRI = np.load(os.path.join(self.fmriPath, "rh_training_fmri.npy"))
        self.lhROIs, self.lhAvgFMRI = getAvgROI(parentDir, subj, self.lhFMRI)
        self.rhROIs, self.rhAvgFMRI = getAvgROI(parentDir, subj, self.rhFMRI, hemi="r")
        self.transform = transform
        if dataIdxs is not None:
            self.imagePaths = self.imagePaths[dataIdxs]
            self.lhFMRI = self.lhFMRI[dataIdxs]
            self.rhFMRI = self.rhFMRI[dataIdxs]
            self.lhAvgFMRI = self.lhAvgFMRI[dataIdxs]
            self.rhAvgFMRI = self.rhAvgFMRI[dataIdxs]
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
        avgLh, avgRh = self.lhAvgFMRI[idx], self.rhAvgFMRI[idx]
        return image, imagePath, torch.tensor(lh, dtype=torch.float32), torch.tensor(rh, dtype=torch.float32), torch.tensor(avgLh, dtype=torch.float32), torch.tensor(avgRh, dtype=torch.float32)


class NSDDatasetClassSubset(Dataset):
    def __init__(self, parentFolderDir: str, imgDataFolderDir: str, subj: int, className: str, idxs: list = None, tsfms = None):
        self.imgPaths, self.trainingIDs = getClassImages(parentFolderDir, imgDataFolderDir, subj, className)
        self.lhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/lh_training_fmri.npy"))[self.trainingIDs]
        self.rhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/rh_training_fmri.npy"))[self.trainingIDs]
        self.tsfms = tsfms
        self.lhROIs, self.lhAvgROIs = getAvgROI(parentFolderDir, subj, self.lhFMRI)
        self.rhROIs, self.rhAvgROIs = getAvgROI(parentFolderDir, subj, self.rhFMRI, "r")
        if idxs is not None:
            self.imgPaths = self.imgPaths[idxs]
            self.lhFMRI = self.lhFMRI[idxs]
            self.rhFMRI = self.rhFMRI[idxs]
            self.lhAvgROIs = self.lhAvgROIs[idxs]
            self.rhAvgROIs = self.rhAvgROIs[idxs]
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, idx):
        img = Image.open(self.imgPaths[idx])
        lh = self.lhFMRI[idx]
        rh = self.rhFMRI[idx]
        lhAvg = self.lhAvgROIs[idx]
        rhAvg = self.rhAvgROIs[idx]
        if self.tsfms:
            img = self.tsfms(img)
        return img, self.imgPaths[idx], torch.from_numpy(lh), torch.from_numpy(rh), torch.tensor(lhAvg, dtype = torch.float32), torch.tensor(rhAvg, dtype = torch.float32)


class BalancedCocoSuperClassDataset(Dataset):
    def __init__(self, parentDir: str, metaDataDir: str, idxs: list = None, tsfms = None):
        superClasses = ['accessory', 'animal', 'appliance', 'electronic', 'food',
       'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports',
       'vehicle']
        self.tsfms = tsfms  
        self.parentDir = parentDir      
        self.imgPaths = np.load(os.path.join(metaDataDir, "newDatasetImagesPath.npy"))
        self.labels = np.load(os.path.join(metaDataDir, "newDatasetImagesLabel.npy"))
        self.labelEncoder = LabelEncoder().fit(superClasses)
        if idxs is not None:
            self.imgPaths = self.imgPaths[idxs]
            self.labels = self.labels[idxs]
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, idx):
        filePath = self.imgPaths[idx].split("C:/Users/josem/Documents/schoolWork/MQP/algonauts2023_transformers#2Leader/algonauts_2023_challenge_data/")[-1]
        img = Image.open(os.path.join(self.parentDir, filePath))
        label = self.labels[idx]
        if self.tsfms:
            img = self.tsfms(img)
        return img, torch.tensor(self.labelEncoder.transform([label]), dtype=torch.long).squeeze()

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

class roiVGGYoloRandomForest(torch.nn.Module):
    def __init__(self, tsfms):
        super(roiVGGYoloRandomForest, self).__init__()
        self.vgg = vgg19(weights = "DEFAULT")
        self.vggConvFeatures = self.vgg.features[:35]
        for params in self.vgg.parameters():
            params.requires_grad = False
        self.yolo = YoloModel("yolov8n.pt")
        for params in self.yolo.parameters():
            params.requires_grad = False
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
            for objectROIPool in objectROIPools:
                # print("objROIP")
                # print(objectROIPool.shape)
                # print(objectROIPool)
                input = torch.flatten(objectROIPool)
                finalFMRIs.append(input)
                indices.append(count)
            # print("fmriPieces")
            # print(f"Shape: ({len(fmriPieces)}, {fmriPieces[0].shape})")
            # print(fmriPieces)
            count += 1
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

def cocoVGGTrain():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    numImages = 13200
    trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

    trainingDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs=trainIdxs,tsfms = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs = validIdxs, tsfms = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)



    numClasses = 12
    model = CocoVGG(numClasses).to(device)
    learningRate = 0.0001
    weightDecay = 1e-4
    optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-4)#,  weight_decay=5e-4
    scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()


    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        numRight = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, label = data
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
        model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, label = data
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                # print(torch.argmax(pred, 1))
                # print(label)
                evalLoss = criterion(pred, label)
                numRight += (torch.argmax(pred, 1) == label).sum().item()
                avgEvalLoss += evalLoss.item()
        print(f"Epoch {epoch} using lr {learningRate} TrainingCE: {avgTrainingLoss / len(trainDataLoader)}, ValidCE: {avgEvalLoss / len(validDataLoader)}, ValidAcc: {numRight / len(validDataset)}, got {numRight} right")
        learningRate = 0.0000001
        if epoch == 0:
            learningRate = 0.0000001
            for g in optim.param_groups:
                g["lr"] = learningRate
        scheduler.step()
        if epoch % 5 == 4:
            # learningRate = 0.00000001
            weightDecay *= 2
            for g in optim.param_groups:
                # g["lr"] = learningRate
                g["weight_decay"] = weightDecay
        model.train()


    torch.save(model.state_dict(), './cocoVGGModel.pth')


def roiTrain():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    numImages = len(getClassImages(parentDir, metaDataDir, subj, "person")[0])
    trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

    trainingDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs=trainIdxs, tsfms = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs = validIdxs, tsfms = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)

    numClasses = 12
    numROIs = len(trainingDataset.lhROIs)
    model = roiVGG(numClasses, "./cocoVGGModel.pth", numROIs, device).to(device)
    learningRate = 0.00001
    optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-2)#,  weight_decay=1e-4
    # scheduler = lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    criterion = torch.nn.MSELoss()


    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        avgTrainingR2Score = 0
        avgEvalR2Score = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, _, _, _, avgFMRI, _ = data
            img = img.to(device)
            avgFMRI = avgFMRI.to(device)
            optim.zero_grad()
            pred = model(img)[1]
            loss = criterion(pred, avgFMRI)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
            # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
            avgTrainingR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
        model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, _, _, _, avgFMRI, _ = data
                img = img.to(device)
                avgFMRI = avgFMRI.to(device)
                pred = model(img)[1]
                # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
                evalLoss = criterion(pred, avgFMRI)
                avgEvalLoss += evalLoss.item()
                # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
                avgEvalR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        print(f"Epoch {epoch} using lr = {learningRate} TrainingMSE: {avgTrainingLoss / len(trainDataLoader)}, ValidMSE: {avgEvalLoss / len(validDataLoader)}, trainR2 = {avgTrainingR2Score / len(trainDataLoader)}, evalR2= {avgEvalR2Score / len(validDataLoader)}")
    # scheduler.step()
    # learningRate = 0.000001
    # if epoch == 10:
    #     first = False
    #     learningRate = 0.0000001
    #     for g in optim.param_groups:
    #         g["lr"] = learningRate
        # if epoch == 5:
        #     learningRate = 0.00000001
        #     for g in optim.param_groups:
        #         g["lr"] = learningRate
        #         # g["weight_decay"] = 1e-3
        # model.train()


    torch.save(model.state_dict(), './cocoVGGROIModel.pth')

def vggYoloTrain():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    yoloTsfms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])


    numImages = len(getClassImages(parentDir, metaDataDir, subj, "person")[0])
    trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

    trainingDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs=trainIdxs, tsfms = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs = validIdxs, tsfms = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)

    # numClasses = 12
    numROIs = len(trainingDataset.lhROIs)
    model = roiVGGYolo(numROIs, yoloTsfms).to(device)
    learningRate = 0.00001
    optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-5)#,  weight_decay=1e-4
    scheduler = lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    criterion = torch.nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        avgTrainingR2Score = 0
        avgEvalR2Score = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, imgPaths, _, _, avgFMRI, _ = data
            img = img.to(device)
            avgFMRI = avgFMRI.to(device)
            optim.zero_grad()
            # print("start")
            # print(imgPaths)
            # print("end")
            pred, indices = model(img, imgPaths)
            avgFMRI = avgFMRI[indices]
            loss = criterion(pred, avgFMRI)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
            # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
            avgTrainingR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
        # model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, imgPaths, _, _, avgFMRI, _ = data
                img = img.to(device)
                avgFMRI = avgFMRI.to(device)
                pred, indices = model(img, imgPaths)
                avgFMRI = avgFMRI[indices]
                # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
                evalLoss = criterion(pred, avgFMRI)
                avgEvalLoss += evalLoss.item()
                # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
                avgEvalR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        scheduler.step()
        print(f"Epoch {epoch} using lr = {learningRate} TrainingMSE: {avgTrainingLoss / len(trainDataLoader)}, ValidMSE: {avgEvalLoss / len(validDataLoader)}, trainR2 = {avgTrainingR2Score / len(trainDataLoader)}, evalR2= {avgEvalR2Score / len(validDataLoader)}")

def vggYoloRandomForest():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    yoloTsfms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])


    trainIdxs = np.load(os.path.join(metaDataDir, f"subj{subj}TrainIdxs.npy"))
    validIdxs = np.load(os.path.join(metaDataDir, f"subj{subj}ValidIdxs.npy"))

    trainingDataset = AlgonautsDataset(parentDir, subj, dataIdxs=trainIdxs, transform = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = AlgonautsDataset(parentDir, subj, dataIdxs=validIdxs, transform = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)

    # numClasses = 12
    model = roiVGGYoloRandomForest(yoloTsfms).to(device)

    features = []
    targets = []

    with torch.no_grad():
        for data in tqdm(trainDataLoader, desc="Train", unit="batch"): 
            img, imgPaths, lhFMRI, _ = data
            img = img.to(device)
            lhFMRI = lhFMRI.to(device)
            rois, indices = model(img, imgPaths)
            lhFMRI = lhFMRI[indices]
            features.extend(rois.cpu().numpy())
            targets.extend(lhFMRI.cpu().numpy())

    features = np.array(features)
    targets = np.array(targets)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=15)
    rf_model.fit(features, targets)

    # Make predictions on the test set
    predictions = rf_model.predict(features)

    # Evaluate the performance
    mse = mean_squared_error(targets, predictions)
    print("Training Mean Squared Error:", mse)

    features = []
    targets = []

    with torch.no_grad():
        for data in tqdm(validDataLoader, desc="Validation", unit="batch"): 
            img, imgPaths, lhFMRI, _ = data
            img = img.to(device)
            lhFMRI = lhFMRI.to(device)
            rois, indices = model(img, imgPaths)
            lhFMRI = lhFMRI[indices]
            features.extend(rois.cpu().numpy())
            targets.extend(lhFMRI.cpu().numpy())

    features = np.array(features)[:3000]
    targets = np.array(targets)[:3000]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(features, targets)

    # Make predictions on the test set
    predictions = rf_model.predict(features)

    # Evaluate the performance
    mse = mean_squared_error(targets, predictions)
    print("Validation Mean Squared Error:", mse)

    with open('randomForestPredictor.pkl','wb') as file:
        pickle.dump(rf_model, file)


# def kFoldVGGYoloMLP():
device = "cuda:1" if torch.cuda.is_available() else "cpu"
subj = 1
parentDir = "./algonauts_2023_challenge_data/"
metaDataDir = "./subjCocoImgData/"

tsfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

yoloTsfms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

trainingDataset = AlgonautsDataset(parentDir, subj,  transform = tsfms)
# validDataset = AlgonautsDataset(parentDir, subj, transform = tsfms)

numROIs = len(trainingDataset.lhAvgFMRI[0])
bestModel = {
    "fold": 1,
    "mse": float('inf'),  # Set to positive infinity initially
    "r2": float('-inf'),  # Set to negative infinity initially
    "params": None  # Initialize with None, will be updated with model state dict
}

k_folds = 5

# Use StratifiedKFold for balanced class distribution in each fold
skf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Loop over folds
for fold, (train_idxs, val_idxs) in enumerate(skf.split(trainingDataset.imagePaths)):
    print(f"Fold {fold + 1}/{k_folds}")
    trainSubset = Subset(trainingDataset, train_idxs)
    trainDataLoader = DataLoader(trainSubset, batch_size = 128, shuffle = True)
    validSubset = Subset(trainingDataset, val_idxs)
    validDataLoader = DataLoader(validSubset, batch_size = 128, shuffle = True)
    # numClasses = 12
    model = roiVGGYolo(numROIs, yoloTsfms).to(device)
    learningRate = 0.00001
    optim = torch.optim.Adam(model.parameters(), learningRate)#,  weight_decay=1e-4
    scheduler = lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    criterion = torch.nn.MSELoss()
    epochs = 15
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        avgTrainingR2Score = 0
        avgEvalR2Score = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, imgPaths, _, _, avgLhFMRI, _ = data
            img = img.to(device)
            avgLhFMRI = avgLhFMRI.to(device)
            optim.zero_grad()
            # print("start")
            # print(imgPaths)
            # print("end")
            pred, indices = model(img, imgPaths)
            avgLhFMRI = avgLhFMRI[indices]
            loss = criterion(pred, avgLhFMRI)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
            # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
            avgTrainingR2Score += r2_score(pred.detach().cpu().numpy(), avgLhFMRI.detach().cpu().numpy())
        # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
        # model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, imgPaths, _, _, avgLhFMRI, _ = data
                img = img.to(device)
                avgLhFMRI = avgLhFMRI.to(device)
                pred, indices = model(img, imgPaths)
                avgLhFMRI = avgLhFMRI[indices]
                # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
                evalLoss = criterion(pred, avgLhFMRI)
                avgEvalLoss += evalLoss.item()
                # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
                avgEvalR2Score += r2_score(pred.detach().cpu().numpy(), avgLhFMRI.detach().cpu().numpy())
        scheduler.step()
        validMse = avgEvalLoss / len(validDataLoader)
        validR2 = avgEvalR2Score / len(validDataLoader)
        print(f"Epoch {epoch} using lr = {learningRate} TrainingMSE: {avgTrainingLoss / len(trainDataLoader)}, ValidMSE: {validMse}, trainR2 = {avgTrainingR2Score / len(trainDataLoader)}, evalR2= {validR2}")
        if validMse < bestModel["mse"]:
            print("BESTMODEL SO FAR")
            bestModel["fold"] = fold
            bestModel["mse"] = validMse
            bestModel["r2"] = validR2
            bestModel["params"] = model.state_dict()  # Update with the current model's state dict

torch.save(bestModel["params"], './5FoldBestModel.pth')
bestModelData = np.array([bestModel["fold"], bestModel["mse"], bestModel["r2"]])
np.save("5FoldBestModelData.npy", bestModelData)



    #model(tsfms(testImg)[None, :, :, :].to("cuda:0"), os.path.join(parentDir, "subj01/training_split/training_images/", "train-0022_nsd-00189"))

