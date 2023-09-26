import os
import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets, plotting
from PIL import Image


def plotAllVertices(args):
    hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
                           hemisphere[0] + 'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_all_vertices,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cool',
        colorbar=False,
        title='All vertices, ' + hemisphere + ' hemisphere'
    )
    plotting.show()


def plotROI(args, hemisphere, roi):
    # hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
    # roi = "EBA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}

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
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_roi,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cool',
        colorbar=False,
        title=roi + ', ' + hemisphere + ' hemisphere'
    )
    plotting.show()


def plotFMRIfromIMG(args, train_img_dir, train_img_list, lh_fmri, rh_fmri):
    img = 0  # @param
    hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}

    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img + 1));

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
                           hemisphere[0] + 'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = lh_fmri[img]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = rh_fmri[img]

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title='All vertices, ' + hemisphere + ' hemisphere'
    )
    plotting.show()


def plotFMRIfromIMGandROI(args, train_img_dir, train_img_list, lh_fmri, rh_fmri, roi, img, hemisphere):
    # img = 0  # @param
    # hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
    # roi = "EBA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}

    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    #plt.figure()
    #plt.axis('off')
    #plt.imshow(train_img)
    #plt.title('Training image: ' + str(img + 1));

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
    challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
                                           hemisphere[0] + 'h.' + roi_class + '_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
                                           hemisphere[0] + 'h.' + roi_class + '_fsaverage_space.npy')
    roi_map_dir = os.path.join(args.data_dir, 'roi_masks',
                               'mapping_' + roi_class + '.npy')
    challenge_roi_class = np.load(challenge_roi_class_dir)
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_roi))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_roi)[0]] = \
            lh_fmri[img, np.where(challenge_roi)[0]]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_roi)[0]] = \
            rh_fmri[img, np.where(challenge_roi)[0]]

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title=roi + ', ' + hemisphere + ' hemisphere'
    )
    plotting.show()
