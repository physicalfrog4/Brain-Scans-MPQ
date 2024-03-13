# Brain-Scans-MQP

The first is what you sent me but I think I needed to change what you were plotting. It was plotting the old surfaces and not the average one
def plotROI(args):
    hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
    roi = "early"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}
    listroi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1",
               "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words",
               "mTL-words"]
    
    # All verticies
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
                           hemisphere[0] + 'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the interactive brain surface map
    fsaverage_response_all = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_response_all[np.where(fsaverage_all_vertices)[0]] = -1
    elif hemisphere == 'right':
        fsaverage_response_all[np.where(fsaverage_all_vertices)[0]] = -1

    for roi in listroi:
        # find the roi class
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
                listroi.index(roi)
            # ^^^ modify this to the number to set the entire ROI to
        elif hemisphere == 'right':
            fsaverage_response[np.where(fsaverage_roi)[0]] = \
                listroi[roi]
                # ^^^ modify this to the number to set the entire ROI to
         
        for idx in (np.where(fsaverage_roi)[0]):
            fsaverage_all_vertices[idx] = listroi.index(roi)
            print(idx, fsaverage_all_vertices[idx])


    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.plot_surf(
        surf_mesh=fsaverage['infl_' + hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_' + hemisphere],
        threshold=1e-14,
        cmap='gist_rainbow',
        colorbar=True,
        title=roi + ', ' + hemisphere + ' hemisphere'
    )
    plotting.show()
