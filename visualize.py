import os
import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets, plotting
from PIL import Image

def plot_predictions(args, lh_correlation, rh_correlation, hemisphere):

    roiList = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies",
               "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
               "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral",
               "parietal"]

    roi_dir = os.path.join(args.data_dir, 'roi_masks',
                           hemisphere[0] + 'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    
    fsaverage_correlation = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = lh_correlation
        # print(fsaverage_correlation[np.where(fsaverage_all_vertices)])
    elif hemisphere == 'right':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = rh_correlation
        # print(fsaverage_correlation[np.where(fsaverage_all_vertices)])

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_correlation,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title='Encoding accuracy, ' + hemisphere + ' hemisphere'
    )
def plot_ROIs(args, lh_fmri, rh_fmri): # still working to make the plots work
    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
    roi = "EBA" #@param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}
    rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces",
             "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal",
               "ventral", "lateral", "parietal"] 
    for roi in rois:
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
        challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
            hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')
        fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
            hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
        roi_map_dir = os.path.join(args.data_dir, 'roi_masks',
            'mapping_'+roi_class+'.npy')
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
            if np.mean(lh_fmri[0,np.where(challenge_roi)[0]]) > 0:
                fsaverage_response[np.where(fsaverage_roi)[0]] = 1
            else:
                fsaverage_response[np.where(fsaverage_roi)[0]] = 0
        elif hemisphere == 'right':
            if np.mean(rh_fmri[0,np.where(challenge_roi)[0]]) > 0:
                fsaverage_response[np.where(fsaverage_roi)[0]] = 1
            else:
                fsaverage_response[np.where(fsaverage_roi)[0]] = 0

        # Create the interactive brain surface map
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title=roi+', '+hemisphere+' hemisphere'
        )
