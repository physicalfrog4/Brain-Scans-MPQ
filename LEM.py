import torch
from numpy import average
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from torchvision.models import vgg19, efficientnet_v2_s, resnet50

class EfficientNetExtractor(torch.nn.Module):
    def __init__(self):
        super(EfficientNetExtractor, self).__init__()
        self.efficientNet = efficientnet_v2_s()
        self.features = self.efficientNet.features[:-1]
        self.finalConvLayer = list(self.efficientNet.features[-1].children())[0]

    def forward(self, img):
        features = self.features(img)
        features = self.finalConvLayer(features)
        return features
    
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


def extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size, device = "cuda:1"):
    # vgg = vgg19(weights="DEFAULT").to(device)
    
    # vggConvFeatures = vgg.features[:35]
    model_layer = "avgpool"

    feature_extractor = Resnet50Extractor().to(device)#create_feature_extractor(vgg, return_nodes=[model_layer])
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
        ft = torch.flatten(ft, 1)
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
        ft = torch.flatten(ft, 1)
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca
