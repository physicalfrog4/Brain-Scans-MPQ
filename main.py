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

# Class to hold arguments
class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)

# Main function
def main():
    # setting up the directories and ARGS
    data_dir = '../MQP/algonauts_2023_challenge_data/'  # Specify the data directory
    parent_submission_dir = '../submission'  # Specify the parent submission directory
    subj = 1  # Specify the subject number

    # Create argument object
    args = argObj(data_dir, parent_submission_dir, subj)

    # Load fMRI data
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    # Specify categories
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

    # Load image directories and lists
    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()

    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))

    print("________ Split Data ________")

    # Split data into training, validation, and test sets
    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    lh_fmri_train = lh_fmri[idxs_train]
    rh_fmri_train = rh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Make Lists ________")

    # Create lists for training, validation, and test images
    train_images = data.makeList(train_img_dir, train_img_list, idxs_train)
    val_images = data.makeList(train_img_dir, train_img_list, idxs_val)
    test_images = data.makeList(test_img_dir, test_img_list, idxs_test)
    torch.cuda.empty_cache()

    print("________ Make Classifications ________")

    # Make image classifications
    lh_classifications = make_classifications(train_images, idxs_train, device)
    rh_classifications = lh_classifications
    lh_classifications_val = make_classifications(val_images, idxs_val, device)
    rh_classifications_val = lh_classifications_val

    torch.cuda.empty_cache()

    print("________ Extract Image Features ________")

    # Transform and extract data features
    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64))

    features_train, features_val, features_test = (
        extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64))

    print("________ LEARN MORE ________")

    # Organize input for training and validation
    LH_train_class, LH_train_FMRI = data.organize_input(lh_classifications, features_train, lh_fmri_train)
    LH_val_class, LH_val_FMRI = data.organize_input(lh_classifications_val, features_val, lh_fmri_val)
    RH_train_class, RH_train_FMRI = data.organize_input(rh_classifications, features_train, rh_fmri_train)
    RH_val_class, RH_val_FMRI = data.organize_input(rh_classifications_val, features_val, rh_fmri_val)

    print("________ Predictions ________")

    # Make predictions
    lh_fmri_val_pred = Predictions(LH_train_class, LH_train_FMRI, LH_val_class)
    rh_fmri_val_pred = Predictions(RH_train_class, RH_train_FMRI, RH_val_class)

    print("________ Analyze Results ________")

    # Analyze prediction results
    analyze_results(LH_val_FMRI, lh_fmri_val_pred)
    analyze_results(RH_val_FMRI, rh_fmri_val_pred)

    print("________ Make Predictions ________")

    # Unnormalize fMRI data and reload original data
    lh_fmri_val_pred = unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
    rh_fmri_val_pred = unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

    print("________ Re-Load Data ________")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Prediction Accuracy ________")

    # Calculate and print prediction accuracy
    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)

    print("________ Visualize Each Class ________")

    # Visualize average predictions for each class
    length = len(words)
    for clss in range(length):
        avg_lh_val = []
        avg_lh_real = []

        avg_rh_val = []
        avg_rh_real = []
     
        for i in range(len(lh_classifications_val)):

            if lh_classifications_val[i][1] == clss:
                avg_lh_val.append(lh_fmri_val_pred[i])
                avg_lh_real.append(lh_fmri_val[i])

            if rh_classifications[i][1] == clss:
                avg_rh_val.append(rh_fmri_val_pred[i])
                avg_rh_real.append(rh_fmri_val[i])

        lh_mean_val = np.mean(avg_lh_val, axis=0)
        rh_mean_val = np.mean(avg_rh_val, axis=0)
        lh_mean_real = np.mean(avg_lh_real, axis=0)
        rh_mean_real = np.mean(avg_rh_real, axis=0)

        # Only look at classes that are observed
        if(len(avg_lh_val) < 10):
            pass
        else:
            print(words[clss])
            
            print("MEAN PRED LH:\n", lh_mean_val)
            plot_predictions(args, lh_mean_val, rh_mean_val, 'left')
            print("MEAN REAL LH:\n", lh_mean_real)
            plot_predictions(args, lh_mean_real, rh_mean_real, 'left')
            cosine = np.dot(lh_mean_val,lh_mean_real)/(norm(lh_mean_val)*norm(lh_mean_real))
            print("Cosine Similarity:", cosine)
        if(len(avg_rh_val) < 10):
            pass
        else:
            print(words[clss])
            print("MEAN PRED RH:\n", rh_mean_val)
            plot_predictions(args, lh_mean_val, rh_mean_val, 'right')
            print("MEAN REAL RH:\n", rh_mean_real)
            plot_predictions(args, lh_mean_real, rh_mean_real, 'right')
            cosine = np.dot(rh_mean_val,rh_mean_real)/(norm(rh_mean_val)*norm(rh_mean_real))
            print("Cosine Similarity:", cosine)
        

    print("________ END ________")


if __name__ == "__main__":
    platform = 'jupyter_notebook'
    device = 'cpu'
    #device = 'cuda:0'
    device = torch.device(device)
    main()
