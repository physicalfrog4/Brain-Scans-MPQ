import torch
from numpy import average
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from torchvision.models import vgg19


def extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size, device = "cuda:1"):
    vgg = vgg19(weights="DEFAULT").to(device)
    vggConvFeatures = vgg.features[:35]
    model_layer = "avgpool"

    feature_extractor = create_feature_extractor(vgg, return_nodes=[model_layer])
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
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting Features"):
        # Send to tensor to device
        d = d.to(device)
        # ...
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        # ft = pca.transform(ft.cuda().detach().numpy())
        features.append(ft)
        val = np.vstack(features)
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
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca
