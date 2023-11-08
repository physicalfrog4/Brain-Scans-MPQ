import random
import torch
from numpy import average
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from PIL import Image
from scipy.stats import pearsonr as corr


def splitData(args, modelGN, modelLR, train_img_list, test_img_list, train_img_dir, test_img_dir, lh_fmri, rh_fmri):
    rand_seed = random.randint(0, 100)
    np.random.seed(rand_seed)

    global lh_correlation
    global rh_correlation

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition, and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the images color channels
    ])
    batch_size = 100  # @param
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform),
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform),
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform),
        batch_size=batch_size
    )
    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]
    del lh_fmri, rh_fmri
    lh_fmri_train = normalize_fmri_data(lh_fmri_train)
    rh_fmri_train = normalize_fmri_data(rh_fmri_train)
    lh_fmri_val = normalize_fmri_data(lh_fmri_val)
    rh_fmri_val = normalize_fmri_data(rh_fmri_val)

    NNclassify(args, modelGN, modelLR, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size,
               lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val)


def NNclassify(args, modelGN, modelLR, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size,
               lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val):
    train_nodes, _ = get_graph_node_names(modelGN)
    model_layer = "model.fc"
    feature_extractor = create_feature_extractor(modelGN, return_nodes=[model_layer])
    pca = fit_pca(feature_extractor, train_imgs_dataloader, batch_size)

    features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
    features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
    features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    del modelGN, pca
    linearMap(args, modelLR, features_train, lh_fmri_train, rh_fmri_train, features_val, features_test, lh_fmri_val,
              rh_fmri_val)
    print("Accuracy Scores")


def linearMap(args, modelLR, features_train, lh_fmri_train, rh_fmri_train, features_val, features_test, lh_fmri_val,
              rh_fmri_val):
    print("Start Linear Map")
    # Fit a linear regression model on the training data
    modelLR = LinearRegression().fit(features_train, lh_fmri_train)

    # Use the fitted model to predict the validation and test fMRI data
    lh_fmri_val_pred = modelLR.predict(features_val)
    lh_fmri_test_pred = modelLR.predict(features_test)

    mae = mean_absolute_error(lh_fmri_val, lh_fmri_val_pred)
    print("Mean Absolute Error on Validation Data:", mae)

    modelLR = LinearRegression().fit(features_train, rh_fmri_train)

    # Use fitted linear regressions to predict the validation and test fMRI data
    rh_fmri_val_pred = modelLR.predict(features_val)
    rh_fmri_test_pred = modelLR.predict(features_test)
    # predict the Accuracy
    predAccuracy(args, lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)


def predAccuracy(args, lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val):
    print("Start PredAccuracy")
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
    # print the results (*100 because I like how it looks)
    print('average lh ', average(lh_correlation) * 100, 'average rh ', average(rh_correlation) * 100)


def extract_features(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        # ft = pca.transform(ft.cuda().detach().numpy())
        features.append(ft)
    return np.vstack(features)


def normalize_fmri_data(data):
    # Perform normalization on the data
    mean_value = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean_value) / std_dev
    return normalized_data


def fit_pca(feature_extractor, dataloader, batch_size):
    torch.device = 'cuda'
    # Define PCA parameters
    pca = IncrementalPCA(batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to('cuda')
        return img
