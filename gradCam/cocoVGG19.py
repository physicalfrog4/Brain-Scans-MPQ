
import os
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from torchvision import transforms

from sklearn.model_selection import train_test_split
from datasets import COCOImgWithLabel, BalancedCocoSuperClassDataset
from models import CocoVGG

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
            # torch.nn.Softmax(dim=1)
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


numImages = len(os.listdir(os.path.join(parentDir, f"subj0{subj}/training_split/training_images/")))
trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

trainingDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs=trainIdxs,tsfms = tsfms)
trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

validDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs = validIdxs, tsfms = tsfms)
validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)



numClasses = 12
model = CocoVGG(numClasses).to(device)
learningRate = 0.0001
optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-4)#,  weight_decay=5e-4
# scheduler = lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()


epochs = 30
first = True
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
    # learningRate = 0.0000001
    # scheduler.step()
    if first:
        first = False
        learningRate = 0.0000001
        for g in optim.param_groups:
            g["lr"] = learningRate
    if epoch == 5:
        learningRate = 0.00000001
        for g in optim.param_groups:
            g["lr"] = learningRate
            g["weight_decay"] = 1e-3
    model.train()


torch.save(model.state_dict(), './cocoVGGModel.pth')

