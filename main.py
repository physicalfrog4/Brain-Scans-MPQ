import os
import sys
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr


def main():
    print("Hello World!")
    if platform == 'jupyter_notebook':
        data_dir = 'C:\GitHub\Brain-Scans-MQP\FMRI-Data'
        parent_submission_dir = 'C:\GitHub\Brain-Scans-MQP\submissiondir'
    subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
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

    hemisphere = 'right'  # @param ['left', 'right'] {allow-input: true}

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
                           hemisphere[0] + 'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    # view1 = plotting.plot_surf(
    #    surf_mesh=fsaverage['infl_' + hemisphere],
    #    surf_map=fsaverage_all_vertices,
    #    bg_map=fsaverage['sulc_' + hemisphere],
    #    threshold=1e-14,
    #    cmap='cool',
    #    colorbar=False,
    #    title="test title"
    #    #title='All vertices, ' + hemisphere + ' hemisphere'
    # )
    # plotting.show()
    roi = "OFA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}

    # Define the ROI class based on the selected ROI
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'

    # Load the ROI brain surface maps
    roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
                                 hemisphere[0] + 'h.' + roi_class + '_fsaverage_space.npy')
    roi_map_dir = os.path.join(args.data_dir, 'roi_masks',
                               'mapping_' + roi_class + '.npy')
    fsaverage_roi_class = np.load(roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view2 = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_roi,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cool',
        colorbar=True,
        title=roi + ', ' + hemisphere + ' hemisphere'
    )
    plotting.show()

    # view2


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
    with zipfile.ZipFile("subj01.zip", "r") as zip_ref:
        zip_ref.extractall("FMRI-Data")


if __name__ == "__main__":
    platform = 'jupyter_notebook'  # @param ['colab', 'jupyter_notebook'] {allow-input: true}
    device = 'cuda'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)
    # uncomment this when first used to unzip the patient data
    # unzipData()
    main()
