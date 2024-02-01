import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import data
from words import makeClassifications, makeMorePred
from data import normalize_fmri_data
from LEM import extract_data_features, linearMap, predAccuracy
from classification import classFMRIfromIMGandROI

def main():
    if platform == 'jupyter_notebook':
        data_dir = '../MQP/algonauts_2023_challenge_data/'
        parent_submission_dir = 'C:\GitHub\Brain-Scans-MQP\submissiondir'
    subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    # args
    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    batch_size = 100
    print("________ Process Data ________")
    # Normalize Data Before Split
    lh_fmri = normalize_fmri_data(lh_fmri)
    print(lh_fmri)
    print("- - - - - - - -")
    rh_fmri = normalize_fmri_data(rh_fmri)
    print(rh_fmri)

    print('LH training fMRI data shape:')
    print(lh_fmri.shape)
    print('(Training stimulus images × LH vertices)')

    print('\nRH training fMRI data shape:')
    print(rh_fmri.shape)
    print('(Training stimulus images × RH vertices)')

    hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
    roi = "OPA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies",
    # "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
    # "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    # {allow-input: true}

    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))
    train_img_file = train_img_list[0]
    print('\nTraining image file name: ' + train_img_file)
    print('\n73k NSD images ID: ' + train_img_file[-9:-4])



    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, batch_size))

    print("________ Image Classification ________")

    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    # change this later to train img dir
    print("________ Create Dataframe ________")
    lh_fmri_train = lh_fmri[idxs_train]
    df_lh_train = data.createDataFrame(idxs_train, lh_fmri_train)
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Create Dataframe For ROI ________")

    df_train = data.dfROI(args, idxs_train, lh_fmri, rh_fmri)
    df_val = data.dfROI(args, idxs_val, lh_fmri, rh_fmri)
    df_test = data.dfROI(args, idxs_test, lh_fmri, rh_fmri)
    print('\nTRAIN\n', df_train, "\nVAL\n", df_val, "\nTEST\n", df_test)

    print("________ Create Dataframe ________")

    df_lh_train = data.createDataFrame(idxs_train, lh_fmri_train)
    df_lh_val = data.createDataFrame(idxs_val, lh_fmri_val)
    rh_fmri_train = rh_fmri[idxs_train]
    df_rh_train = data.createDataFrame(idxs_train, rh_fmri_train)
    rh_fmri_val = rh_fmri[idxs_val]
    df_rh_val = data.createDataFrame(idxs_val, rh_fmri_val)
    torch.cuda.empty_cache()

    print("________ Make Classifications ________")
    modelYOLO = YOLO('yolov8n-cls.pt')
    modelYOLO.to("cuda:1")
    lh_classifications_val = makeClassifications(idxs_val, train_img_list, train_img_dir)
    rh_classifications_val = lh_classifications_val
    torch.cuda.empty_cache()
    lh_classifications = makeClassifications(idxs_train, train_img_list, train_img_dir)
    rh_classifications = lh_classifications
    torch.cuda.empty_cache()
    # print(lh_classifications)

    print("________ Combine Dataframes ________")
    lh_train = pd.concat([lh_classifications, df_lh_train], axis=1)
    lh_train = lh_train.drop(["Name", "Num"], axis=1)
    lh_train = lh_train.fillna(-1)
    print(lh_train)
    # lh_train = lh_train[lh_train['Class'].notna()]

    rh_train = pd.concat([rh_classifications, df_rh_train, ], axis=1)
    rh_train = rh_train.drop(["Name", "Num"], axis=1)
    rh_train = rh_train.fillna(-1)
    # rh_train = rh_train[rh_train['Class'].notna()]

    rh_classifications_val = pd.concat([rh_classifications_val, df_rh_val], axis=1)
    rh_classifications_val = rh_classifications_val.drop(["Name", "Num"], axis=1)
    # rh_val = rh_classifications_val[rh_classifications_val['Class'].notna()]
    rh_val = rh_classifications_val.fillna(-1)

    lh_classifications_val = pd.concat([lh_classifications_val, df_lh_val], axis=1)
    lh_classifications_val = lh_classifications_val.drop(["Name", "Num"], axis=1)
    lh_val = lh_classifications_val.fillna(-1)
    # lh_val = lh_classifications_val[lh_classifications_val['Class'].notna()]

    print(len(lh_train))
    print(len(rh_train))
    print(len(lh_val))
    print(len(rh_val))

    lh_fmri_val_pred = makeMorePred(lh_train, lh_val)
    rh_fmri_val_pred = makeMorePred(rh_train, rh_val)

    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)
    print(lh_correlation, rh_correlation)

    print("________ End ________")
    exit()
    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]
    del lh_fmri, rh_fmri


    print("________ MOBILE NET ________")
    # Google Net Model
    modelGN = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    modelGN.to('cuda')  # send the model to the chosen device ('cpu' or 'cuda')
    modelGN.eval()  # set the model to evaluation mode, since you are not training it

    features_train, features_val, features_test = (
        extract_data_features(modelGN, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size))
    del modelGN
    lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_test_pred = (
        linearMap(features_train, lh_fmri_train, rh_fmri_train, features_val, features_test, lh_fmri_val, rh_fmri_val))

    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)


class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)

        # Create the submission directory if not existing
        # if not os.path.isdir(self.subject_submission_dir):
        # os.makedirs(self.subject_submission_dir)


if __name__ == "__main__":
    platform = 'jupyter_notebook'  # @param ['colab', 'jupyter_notebook'] {allow-input: true}
    device = 'cuda'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)

    main()
