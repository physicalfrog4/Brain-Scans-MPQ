import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split



def classFMRIfromIMGandROI(args, train_img_dir, train_img_list, lh_fmri, rh_fmri, ImgClasses, length):
    excelData = []
    for img in range(len(length)):
        hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
        listroi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA",
                   "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
                   "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral",
                   "parietal"]
        for roi in listroi:
            # Load the image
            img_dir = os.path.join(train_img_dir, train_img_list[img])
            train_img = Image.open(img_dir).convert('RGB')

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
            accuracy = np.mean(fsaverage_response[np.where(fsaverage_roi)[0]])
            if accuracy > 0.001:
                excelData.append([ImgClasses[img], roi, accuracy])

    # print(excelData)
    df = pd.DataFrame(excelData, columns=['Name', 'ROI', 'Accuracy'])
    #df.to_excel('class_data.xlsx', index=False)

    # print(df['ROI'].values, df['Accuracy'])
    # plt.hist(excelData['Accuracy'], color='lightgreen', ec='black', bins=15)
    # plt.show()
    # feature encoding

    # Encode categorical variables like 'Name' and 'ROI' using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Name', 'ROI'], drop_first=True)

    # Split the data into features (X) and target variable (y)
    X = df_encoded.drop('Accuracy', axis=1)
    y = df_encoded['Accuracy']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")
    accuracy_score = model.score(X_test, y_test)
    print("accuracy score", accuracy_score)

    normAcc = normalize_fmri_data2(df['Accuracy'])
    print(normAcc)
    df['Accuracy'] = normAcc
    print(df['Accuracy'])
    print(df)
    return df


def normalize_fmri_data2(data):
    mean_value = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean_value) / std_dev
    return normalized_data
