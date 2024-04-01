# %%
import os
import numpy as np
import torch
from nilearn import plotting
import data
import visualize
from words import make_classifications, Predictions
from data import normalize_fmri_data, unnormalize_fmri_data
from LEM import extract_data_features, predAccuracy
from numpy.linalg import norm


# %%
class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)


# %%
platform = 'jupyter_notebook'

device = "cuda:1"
batch_size = 32
torch.cuda.set_device(device)
# setting up the directories and ARGS
data_dir = '/home/vislab-001/Documents/algonauts_2023_challenge_data/'
parent_submission_dir = '../submission'
subj = 1 # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
# args

args = argObj(data_dir, parent_submission_dir, subj)
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
            'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing', 
             'computer', 'fruit', 'vegetable', 'tool']


# %%
print("________ Process Data ________")

# Normalize Data Before Split
lh_fmri, lh_data_min, lh_data_max = normalize_fmri_data(lh_fmri)
print(lh_fmri)
print("- - - - - - - -")
rh_fmri, rh_data_min, rh_data_max = normalize_fmri_data(rh_fmri)
print(rh_fmri)

print('LH training fMRI data shape:')
print(lh_fmri.shape)
print('(Training stimulus images × LH vertices)')

print('\nRH training fMRI data shape:')
print(rh_fmri.shape)
print('(Training stimulus images × RH vertices)')

train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')
train_img_list = os.listdir(train_img_dir)
train_img_list.sort()
train_img_list = train_img_list[:2500]
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()

print('\nTraining images: ' + str(len(train_img_list)))
print('\nTest images: ' + str(len(test_img_list)))
train_img_file = train_img_list[0]
print('\nTraining image file name: ' + train_img_file)
print('\n73k NSD images ID: ' + train_img_file[-9:-4])

# %%
print("________ Split Data ________")

idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
lh_fmri_train = lh_fmri[idxs_train]
rh_fmri_train = rh_fmri[idxs_train]
lh_fmri_val = lh_fmri[idxs_val]
rh_fmri_val = rh_fmri[idxs_val]
lh_fmri_test = lh_fmri[idxs_test]
rh_fmri_test = rh_fmri[idxs_test]

# %%
print("________ Make Lists ________")

train_images = data.makeList(train_img_dir, train_img_list, idxs_train)
val_images = data.makeList(train_img_dir, train_img_list, idxs_val)
test_images = data.makeList(test_img_dir, test_img_list, idxs_test)
torch.cuda.empty_cache()

# %%
print("________ Make Classifications ________")
lh_classifications = make_classifications(train_images, idxs_train, device)
rh_classifications = lh_classifications
lh_classifications_val = make_classifications(val_images, idxs_val, device)
rh_classifications_val = lh_classifications_val
lh_classifications_test = make_classifications(test_images, idxs_test, device)
rh_classifications_test = lh_classifications_test

torch.cuda.empty_cache()

# %%
import torch
from numpy import average
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from torchvision.models import vgg19, efficientnet_v2_s, inception_v3, resnet50, resnet101, wide_resnet50_2, wide_resnet101_2

class EfficientNetThroughLastConvExtractor(torch.nn.Module):
    def __init__(self):
        super(EfficientNetThroughLastConvExtractor, self).__init__()
        self.efficientNet = efficientnet_v2_s()
        self.features = self.efficientNet.features[:-1]
        self.finalConvLayer = list(self.efficientNet.features[-1].children())[0]

    def forward(self, img):
        features = self.features(img)
        features = self.finalConvLayer(features)
        return features

    def __str__(self):
        return ""
    
    def __repr__(self):
        return ""


class EfficientNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientNet = efficientnet_v2_s()

    def forward(self, img):
        features = self.efficientNet.features(img)
        return features

    def __str__(self):
        return ""
    
    def __repr__(self):
        return ""

class InceptionEXtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionEXtractor, self).__init__()
        self.inception = inception_v3(weights="DEFAULT")

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = self.inception.avgpool(x)

        return x

class Resnet50Extractor(torch.nn.Module):
    def __init__(self):
        super(Resnet50Extractor, self).__init__()
        self.resnet = resnet50(weights="DEFAULT")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x

class Resnet101Extractor(torch.nn.Module):
    def __init__(self):
        super(Resnet101Extractor, self).__init__()
        self.resnet = resnet101(weights="DEFAULT")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x

class WideResnet502(torch.nn.Module):
    def __init__(self):
        super(WideResnet502, self).__init__()
        self.resnet = wide_resnet50_2(weights="DEFAULT")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x

class WideResnet1012(torch.nn.Module):
    def __init__(self):
        super(WideResnet1012, self).__init__()
        self.resnet = wide_resnet101_2(weights="DEFAULT")

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x


def extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size, device = "cuda:1"):
    # vgg = vgg19(weights="DEFAULT").to(device)
    # vggConvFeatures = vgg.features[:35]
    model_layer = "avgpool"

    feature_extractor = WideResnet502().to(device)#create_feature_extractor(vgg, return_nodes=[model_layer])
    pca = fit_pca(feature_extractor, train_imgs_dataloader, batch_size, device)

    features_train = extract_features(feature_extractor, train_imgs_dataloader, pca, device)
    features_val = extract_features(feature_extractor, val_imgs_dataloader, pca, device)
    features_test = extract_features(feature_extractor, test_imgs_dataloader, pca, device)

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    del pca
    return features_train, features_val, features_test



def predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val):
    print("Start PredAccuracy")
    print("\npredicted\n", lh_fmri_val_pred, "\nactual\n", lh_fmri_val)

    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])

    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:, v], lh_fmri_val[:, v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])

    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:, v], rh_fmri_val[:, v])[0]
        
    print('average lh ', average(lh_correlation) * 100, 'average rh ', average(rh_correlation) * 100)
    return lh_correlation, rh_correlation


def extract_features(feature_extractor, dataloader, pca, device):
    features = []
    pcaMean = torch.tensor(pca.mean_, dtype=torch.float32).to(device)
    pcaMatrix = torch.tensor(pca.components_, dtype=torch.float32).to(device)
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting Features"):
        # Send to tensor to device
        d = d.to(device)
        # ...
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.flatten(ft, 1)
        # Apply PCA transform
        ft = torch.matmul(ft - pcaMean, pcaMatrix.T)
        # ft = pca.transform(ft.cuda().detach().numpy())
        features.append(ft.cpu().detach().numpy())
        # val = np.vstack(features)
        # print(val)
    return np.vstack(features)


def fit_pca(feature_extractor, dataloader, batch_size, device):
    # Define PCA parameters
    pca = IncrementalPCA(batch_size=batch_size)
    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader), desc="PCA"):
        # Send to tensor to device
        d = d.to(device)
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.flatten(ft, 1)
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca

# %%
print("________ Extract Image Features ________")

train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
    data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, batch_size))


# %%
# features_train, features_val, features_test = (
#     extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size, device))

feature_extractor = WideResnet502().to(device)#create_feature_extractor(vgg, return_nodes=[model_layer])

# Define PCA parameters
pca = IncrementalPCA(batch_size=batch_size)
# Fit PCA to batch
for _, d in tqdm(enumerate(train_imgs_dataloader), total=len(train_imgs_dataloader), desc="PCA"):
    # Send to tensor to device
    d = d.to(device)
    # Extract features
    ft = feature_extractor(d)
    # Flatten the features
    ft = torch.flatten(ft, 1)
    # Fit PCA to batch
    pca.partial_fit(ft.detach().cpu().numpy())


# features_train = extract_features(feature_extractor, train_imgs_dataloader, pca, device)
# features_val = extract_features(feature_extractor, val_imgs_dataloader, pca, device)
# features_test = extract_features(feature_extractor, test_imgs_dataloader, pca, device)

# print('\nTraining images features:')
# print(features_train.shape)
# print('(Training stimulus images × PCA features)')

# print('\nValidation images features:')
# print(features_val.shape)
# print('(Validation stimulus images × PCA features)')

# print('\nTest images features:')
# print(features_test.shape)
# print('(Test stimulus images × PCA features)')


# %%
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
from torchvision import transforms

class AlisModel(torch.nn.Module):
    def __init__(self, feature_extractor, pca, featureSize, numROIs):
        print("Super")
        super(AlisModel, self).__init__()
        print("ft")
        self.feature_extractor = 1
        print("mean")
        self.pcaMean = 1#torch.tensor(pca.mean_, dtype=torch.float32)
        print("mat")
        self.pcaMatrix = 2#torch.tensor(pca.components_, dtype=torch.float32)
        print("linear")
        self.linear = torch.nn.Linear(featureSize, numROIs)
        print("done")

    def forward(self, x): #x is original image
        x = self.feature_extractor(x) # x is image features
        print(x.shape)
        print(self.pcaMean.shape)
        print(self.pcaMatrix.shape)
        x = torch.matmul(x - self.pcaMean, self.pcaMatrix.T) #now x is the transformed features
        x = self.linear(x)
        return x
    
#currently using
#Creates dataset with all training images for a specific subject 
class AlgonautsDataset(Dataset):
    def __init__(self, parentDir: str, subj: int, dataIdxs: list = None, transform = None):
        self.imagesPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_images/")
        self.fmriPath = os.path.join(parentDir, f"subj0{subj}/training_split/training_fmri/")
        self.imagePaths = np.array(os.listdir(self.imagesPath))
        self.lhFMRI = np.load(os.path.join(self.fmriPath, "lh_training_fmri.npy"))
        self.rhFMRI = np.load(os.path.join(self.fmriPath, "rh_training_fmri.npy"))
        self.transform = transform
        if dataIdxs is not None:
            self.imagePaths = self.imagePaths[dataIdxs]
            self.lhFMRI = self.lhFMRI[dataIdxs]
            self.rhFMRI = self.rhFMRI[dataIdxs]
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
        return image, imagePath, torch.tensor(lh, dtype=torch.float32), torch.tensor(rh, dtype=torch.float32)




# %%
model = AlisModel(feature_extractor, "pca", 100352, 19004)#ft.shape[1], lh_fmri.shape[1]).to(device)


# %%



