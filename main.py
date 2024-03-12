# Import necessary libraries
import os
import numpy as np
import torch
from nilearn import plotting
import data
import visualize
from words import make_classifications, Predictions
from data import normalize_fmri_data, unnormalize_fmri_data, analyze_results
from LEM import extract_data_features, predAccuracy
from visualize import plot_predictions
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.neural_network import MLPRegressor


def main():
    # setting up the directories and ARGS
    data_dir = '../MQP/algonauts_2023_challenge_data/'
    parent_submission_dir = '../submission'
    subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}

    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
             'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing',
             'safety', 'luggage', 'computer', 'fruit', 'vegetable', 'tool']

    print("________ Process Data ________")

    # Normalize Data Before Split
    lh_fmri, lh_data_min, lh_data_max = normalize_fmri_data(lh_fmri)
    rh_fmri, rh_data_min, rh_data_max = normalize_fmri_data(rh_fmri)

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
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()

    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))

    print("________ Split Data ________")

    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    lh_fmri_train = lh_fmri[idxs_train]
    rh_fmri_train = rh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Make Lists ________")

    train_images = data.makeList(train_img_dir, train_img_list, idxs_train)
    val_images = data.makeList(train_img_dir, train_img_list, idxs_val)
    test_images = data.makeList(test_img_dir, test_img_list, idxs_test)
    torch.cuda.empty_cache()

    print("________ Make Classifications ________")

    lh_classifications = make_classifications(train_images, idxs_train, device)
    rh_classifications = lh_classifications
    lh_classifications_val = make_classifications(val_images, idxs_val, device)
    rh_classifications_val = lh_classifications_val

    torch.cuda.empty_cache()

    print("________ Extract Image Features ________")

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64))

    features_train, features_val, features_test = (
        extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64))

    print("________ LEARN MORE ________")

    LH_train_class, LH_train_FMRI = data.organize_input(lh_classifications, features_train, lh_fmri_train)
    LH_val_class, LH_val_FMRI = data.organize_input(lh_classifications_val, features_val, lh_fmri_val)
    RH_train_class, RH_train_FMRI = data.organize_input(rh_classifications, features_train, rh_fmri_train)
    RH_val_class, RH_val_FMRI = data.organize_input(rh_classifications_val, features_val, rh_fmri_val)

    print("________ Predictions ________")

    lh_fmri_val_pred = Predictions(LH_train_class, LH_train_FMRI, LH_val_class, LH_val_FMRI)
    rh_fmri_val_pred = Predictions(RH_train_class, RH_train_FMRI, RH_val_class, RH_val_FMRI)

    print("________ Analyze Results ________")

    analyze_results(LH_val_FMRI, lh_fmri_val_pred)
    analyze_results(RH_val_FMRI, rh_fmri_val_pred)

    # Make predictions
    LR = LinearRegression()
    DT = tree.DecisionTreeRegressor()
    MLP = MLPRegressor()
    print("________ Linear Regression Predictions ________")
    lh_fmri_val_pred = Predictions(LH_train_class, LH_train_FMRI, LH_val_class, LR)
    rh_fmri_val_pred = Predictions(RH_train_class, RH_train_FMRI, RH_val_class, LR)

    
    print("________ Decision Tree Predictions ________")
    lh_fmri_val_pred = Predictions(LH_train_class, LH_train_FMRI, LH_val_class, DT)
    rh_fmri_val_pred = Predictions(RH_train_class, RH_train_FMRI, RH_val_class,DT)

    
    print("________ MLP Predictions ________")
    lh_fmri_val_pred = Predictions(LH_train_class, LH_train_FMRI, LH_val_class, MLP)
    rh_fmri_val_pred = Predictions(RH_train_class, RH_train_FMRI, RH_val_class, MLP)

    print("________ Re-Load Data ________")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Prediction Accuracy ________")

    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)

    print("________ Visualize Each Class ________")

    length = len(words)
    for clss in range(length):
        avg_lh_pred = []
        avg_lh_real = []
        avg_rh_pred = []
        avg_rh_real = []
        for i in range(len(lh_classifications_val)):

            if lh_classifications_val[i][1] == clss:
                avg_lh_pred.append(lh_fmri_val_pred[i])
                avg_lh_real.append(lh_fmri_val[i])

            if rh_classifications[i][1] == clss:
                avg_rh_pred.append(rh_fmri_val_pred[i])
                avg_rh_real.append(lh_fmri_val[i])
        # Only look at classes that are observed 
        if(len(avg_lh_pred)== 0):
            pass
        else:
            lh = np.mean(avg_lh_pred, axis=0)
            rh = np.mean(avg_rh_pred, axis=0)
            #print("MEAN PRED LH:\n", lh)
            #print("MEAN PRED RH:\n", rh)
            # visualize.plot_predictions(args, lh, rh)
            lh2 = np.mean(avg_lh_real, axis=0)
            rh2 = np.mean(avg_rh_real, axis=0)
            #print("MEAN REAL LH:\n", lh2)
            #print("MEAN REAL RH:\n", rh2)
            # visualize.plot_predictions(args, lh2, rh2)
            # plotting.show()
            corr = np.corrcoef(avg_lh_pred, avg_lh_real)
            print(len(avg_lh_pred))
            print("Corre ", np.mean(corr))
            cosine = np.dot(lh,lh2)/(norm(lh)*norm(lh2))
            print("Cosine Similarity:", cosine)
            torch.cuda.empty_cache()

    print("________ END ________")


class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)


if __name__ == "__main__":
    platform = 'jupyter_notebook'
    device = 'cuda:0'
    device = torch.device(device)
    main()
