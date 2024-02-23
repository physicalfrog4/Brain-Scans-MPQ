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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # setting up the directories and ARGS
    data_dir = ''#../MQP/algonauts_2023_challenge_data/'
    parent_submission_dir = '../submission'
    subj = 5 # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    # args

    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    device = 'cuda:1'

    words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
                'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing', 
                'computer', 'fruit', 'vegetable', 'tool']

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
    # train_img_list = train_img_list[:500]
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()

    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))
    train_img_file = train_img_list[0]
    print('\nTraining image file name: ' + train_img_file)
    print('\n73k NSD images ID: ' + train_img_file[-9:-4])

    print("________ Split Data ________")

    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    lh_fmri_train = lh_fmri[idxs_train]
    rh_fmri_train = rh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]
    lh_fmri_test = lh_fmri[idxs_test]
    rh_fmri_test = rh_fmri[idxs_test]

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
    lh_classifications_test = make_classifications(test_images, idxs_test, device)
    rh_classifications_test = lh_classifications_test

    torch.cuda.empty_cache()

    print("________ Extract Image Features ________")

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64))

    features_train, features_val, features_test = (
        extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64))

    print("________ Organize Input________")

    dftrainL, dftrainFL = data.organize_input(lh_classifications, features_train, lh_fmri_train)
    dftrainR, dftrainFR = data.organize_input(rh_classifications, features_train, rh_fmri_train)

    dfvalL, dfvalFL = data.organize_input(lh_classifications_val, features_val, lh_fmri_val)
    dfvalR, dfvalFR = data.organize_input(rh_classifications_val, features_val, rh_fmri_val)

    dftestL, dftestFL = data.organize_input(lh_classifications_test, features_test, lh_fmri_test)
    dftestR, dftestFR = data.organize_input(rh_classifications_test, features_test, rh_fmri_test)

    print("________ Predictions ________")

    lh_fmri_val_pred = Predictions(dftrainL, dftrainFL, dfvalL)
    rh_fmri_val_pred = Predictions(dftrainR, dftrainFR, dfvalR)

    train = dftrainL.to_numpy()
    train_fmri = dftrainFL.to_numpy()
    val = dfvalL.to_numpy()
    val_fmri = dfvalFL.to_numpy()

    print(val_fmri, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", lh_fmri_val_pred)
    linear_regression_mse = mean_squared_error(val_fmri, lh_fmri_val_pred)
    print(f'Random Forest Mean Squared Error: {linear_regression_mse}')
    linear_regression_mae = mean_absolute_error(val_fmri, lh_fmri_val_pred)
    print(f'Random Forest Mean Absolute Error: {linear_regression_mae}')
    linear_regression_r2 = r2_score(val_fmri, lh_fmri_val_pred)
    print(f'Random Forest Mean R 2 Score: {linear_regression_r2}')

    print("________ Make Predictions ________")

    lh_fmri_val_pred = unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
    rh_fmri_val_pred = unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

    print("________ Re-Load Data ________")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Prediction Accuracy ________")

    # lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)

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

        lh = np.mean(avg_lh_pred, axis=0)
        rh = np.mean(avg_rh_pred, axis=0)

        print("MEAN PRED LH:\n", lh)
        print("MEAN PRED RH:\n", rh)
        visualize.plot_predictions(args, lh, rh)
        lh2 = np.mean(avg_lh_real, axis=0)
        rh2 = np.mean(avg_rh_real, axis=0)

        print("MEAN REAL LH:\n", lh2)
        print("MEAN REAL RH:\n", rh2)
        visualize.plot_predictions(args, lh2, rh2)
        plotting.show()

        corr = np.corrcoef(avg_lh_pred, avg_lh_real)
        print("Corre ", np.mean(corr))
        torch.cuda.empty_cache()

    print("________ END ________")
             
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
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()

    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))
    train_img_file = train_img_list[0]
    print('\nTraining image file name: ' + train_img_file)
    print('\n73k NSD images ID: ' + train_img_file[-9:-4])

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

    lh_classifications_val = make_classifications(val_images, idxs_val, device)
    rh_classifications_val = lh_classifications_val
    lh_classifications = make_classifications(train_images, idxs_train, device)
    rh_classifications = lh_classifications
    torch.cuda.empty_cache()

    print("________ Extract Image Features ________")

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64))

    features_train, features_val, features_test = (
        extract_data_features(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64))

    print("________ LEARN MORE ________")

    dftrainL, dftrainFL = data.organize_input(lh_classifications, features_train, lh_fmri_train)
    dfvalL, dfvalFL = data.organize_input(lh_classifications_val, features_val, lh_fmri_val)
    dftrainR, dftrainFR = data.organize_input(rh_classifications, features_train, rh_fmri_train)
    dfvalR, dfvalFR = data.organize_input(rh_classifications_val, features_val, rh_fmri_val)

    print("________ Predictions ________")

    lh_fmri_val_pred = Predictions(dftrainL, dftrainFL, dfvalL, dfvalFL)
    rh_fmri_val_pred = Predictions(dftrainR, dftrainFR, dfvalR, dfvalFR)

    print("________ Make Predictions ________")

    lh_fmri_val_pred = unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
    rh_fmri_val_pred = unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

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

        lh = np.mean(avg_lh_pred, axis=0)
        rh = np.mean(avg_rh_pred, axis=0)

        print("MEAN PRED LH:\n", lh)
        print("MEAN PRED RH:\n", rh)
        visualize.plot_predictions(args, lh, rh)
        lh2 = np.mean(avg_lh_real, axis=0)
        rh2 = np.mean(avg_rh_real, axis=0)

        print("MEAN REAL LH:\n", lh2)
        print("MEAN REAL RH:\n", rh2)
        visualize.plot_predictions(args, lh2, rh2)
        plotting.show()

        corr = np.corrcoef(avg_lh_pred, avg_lh_real)
        print("Corre ", np.mean(corr))
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
