from nilearn import plotting
import os
import zipfile
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO

import LEM
# import classes
import LEM2
import visualize
from words import moreWords


def main():
    if platform == 'jupyter_notebook':
        data_dir = '../MQP/algonauts_2023_challenge_data/'
        parent_submission_dir = 'C:\GitHub\Brain-Scans-MQP\submissiondir'
    subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    # global args
    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
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

    # visualize.plotAllVertices(args)
    # visualize.plotROI(args, hemisphere, roi)
    # plot the ROI
    # visualize.plotROI(args, 'right', roi)
    # plotting.show()

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

    #modelYOLO = YOLO('yolov8n-cls.pt')
    # predicts what the image is based on the preloaded YOLO model.
    #image_results = modelYOLO.predict(test_img_dir)

    #del modelYOLO
    # take the predictions and categorizes them
    #ImgClasses = moreWords(image_results)
    #print(ImgClasses)

    img = 0

    # visualize.plotFMRIfromIMG(args, train_img_dir, train_img_list, lh_fmri, rh_fmri)
    # PLot ROI from FRMI and IMG
    #visualize.plotFMRIfromIMGandROI(args, train_img_dir, train_img_list, lh_fmri, rh_fmri, roi, img, hemisphere)
    #plotting.show()
    # 2
    print("________ MOBILE NET ________")
    modelGN = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    modelGN.to('cuda')  # send the model to the chosen device ('cpu' or 'cuda')
    modelGN.eval()  # set the model to evaluation mode, since you are not training it
    modelLR = LinearRegression()
    LEM.splitData(args, modelGN, modelLR, train_img_list, test_img_list, train_img_dir, test_img_dir, lh_fmri, rh_fmri)
    torch.cuda.empty_cache()
    #print("________ GOOGLE NET ________")
    #LEM2.splitData(args, train_img_list, test_img_list, train_img_dir, test_img_dir, lh_fmri, rh_fmri)

    # LEM.alexnet(train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader,batch_size)

    # visualize.anotherOne(args, LEM.lh_correlation, LEM.rh_correlation)
    # visualize.AccuracyROI(args, LEM.lh_correlation, LEM.rh_correlation)


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


def unzipData():
    with zipfile.ZipFile("daradir/subj02.zip", "r") as zip_ref:
        zip_ref.extractall("FMRI-Data")


if __name__ == "__main__":
    platform = 'jupyter_notebook'  # @param ['colab', 'jupyter_notebook'] {allow-input: true}
    device = 'cuda'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)
    # uncomment this when first used to unzip the patient data
    # unzipData()
    main()