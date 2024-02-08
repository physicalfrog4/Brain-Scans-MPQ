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
    classes = sorted(pklData["classLabel"].unique())
    labelEncoder = LabelEncoder().fit(classes)
    return pklData["classLabel"], labelEncoder



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
    