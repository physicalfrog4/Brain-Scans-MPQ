import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def getFileNames(parentDir: str, subj: int):
    imgsPath = f"subj0{subj}/training_split/training_images/"
    filesDir = os.path.join(parentDir,imgsPath)
    fileNames = os.listdir(filesDir)
    imgPaths = np.array([os.path.join(parentDir, imgsPath, img) for img in fileNames])
    return imgPaths


def getImgLabels(metaDataDir: str, subj: int):
    filePath = os.path.join(metaDataDir, f"subj0{subj}ImgData.pkl")
    pklData = pd.read_pickle(filePath)
    classes = sorted(pklData["superClassLabel"].unique())
    labelEncoder = LabelEncoder().fit(classes)
    return pklData["superClassLabel"], labelEncoder

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

class NSDDatasetClassSubset(Dataset):
    def __init__(self, parentFolderDir: str, imgDataFolderDir: str, subj: int, className: str, idxs: list = None, tsfms = None):
        self.imgPaths, self.trainingIDs = getClassImages(parentFolderDir, imgDataFolderDir, subj, className)
        self.lhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/lh_training_fmri.npy"))[self.trainingIDs]
        self.rhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/rh_training_fmri.npy"))[self.trainingIDs]
        self.tsfms = tsfms
        self.lhROIs, self.lhAvgROIs = getAvgROI(parentFolderDir, subj, self.lhFMRI)
        self.rhROIs, self.rhAvgROIs = getAvgROI(parentFolderDir, subj, self.rhFMRI, "r")
        if idxs is not None:
            print("not null")
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

        



class COCOImgWithLabel(Dataset):
    def __init__(self, parentDir: str, metaDataDir: str, subj: int, idxs: list = None, tsfms = None):
        self.tsfms = tsfms        
        self.imgPaths = getFileNames(parentDir, subj)
        self.labels, self.labelEncoder = getImgLabels(metaDataDir, subj)
        if idxs is not None:
            self.imgPaths = self.imgPaths[idxs]
            self.labels = self.labels[idxs].reset_index(drop=True)
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, idx):
        img = Image.open(self.imgPaths[idx])
        label = self.labels[idx]
        if self.tsfms:
            img = self.tsfms(img)
        return img, torch.tensor(self.labelEncoder.transform([label]), dtype=torch.long).squeeze()

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

# testImg, testLabel = trainingDataset.__getitem__(0)
# testImg = testImg.to(device)
# model(testImg[None, :, :, :])
# torch.argmax(model(testImg[None, :, :, :]))
    