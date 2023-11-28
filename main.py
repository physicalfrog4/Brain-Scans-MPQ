import os
import numpy as np
import torch
import data
from words import wordClassifier
from data import normalize_fmri_data
from LEM import extract_data_features, linearMap, predAccuracy
from classification import classFMRIfromIMGandROI


def main():
    # setting up the directories and ARGS
    data_dir = ''
    parent_submission_dir = '../submission'
    subj = 8
    args = argObj(data_dir, parent_submission_dir, subj)

    # FMRI Data

    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    print(fmri_dir)
    print(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    print('LH training fMRI data shape:')
    print(lh_fmri.shape)
    print('(Training stimulus images × LH vertices)')

    print('\nRH training fMRI data shape:')
    print(rh_fmri.shape)
    print('(Training stimulus images × RH vertices)')
    ## this works

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
    # train_img_list = train_img_list[: 1500]
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()

    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))
    train_img_file = train_img_list[0]
    print('\nTraining image file name: ' + train_img_file)
    print('\n73k NSD images ID: ' + train_img_file[-9:-4])

    print("________ Mobile Net ________")

    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)
    # change this later to train img dir
    ImgClasses = wordClassifier(test_img_dir)
    batch_size = 100
    length = len(idxs_train)

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, batch_size))

    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]

    del lh_fmri, rh_fmri

    # Normalize the Data
    lh_fmri_train = normalize_fmri_data(lh_fmri_train)
    rh_fmri_train = normalize_fmri_data(rh_fmri_train)
    lh_fmri_val = normalize_fmri_data(lh_fmri_val)
    rh_fmri_val = normalize_fmri_data(rh_fmri_val)

    # uncomment this one
    print("here")

    print(length)
    image_class_data = classFMRIfromIMGandROI(args, train_img_dir, train_img_list, lh_fmri_train, rh_fmri_train,
                                              ImgClasses, length)
    exit()

    # Google Net Model
    modelGN = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    modelGN.to('cpu')  # send the model to the chosen device ('cpu' or 'cuda')
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
    device = 'cpu'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)
    # print(torch.cuda.is_available())
    # uncomment this when first used to unzip the patient data
    # nzipData()
    main()
