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
    
