import os

import gensim.downloader as api
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from nltk.corpus import words
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from ultralytics import YOLO

from data import makeList


def wordClassifier(train_img_list):
    w2v = api.load("word2vec-google-news-300")
    # Uncomment this if you want the word classifier
    # not pretrained
    # modelYOLO = YOLO('yolov8n.pt')
    # results = modelYOLO.train(data='coco.yaml', epochs=1, imgsz=640)
    # modelYOLO.val()

    # pretrained

    modelYOLO = YOLO('yolov8n-cls.pt')
    image_results = modelYOLO.predict(train_img_list, stream=True)
    modelYOLO.to('cuda')
    results = []

    # Perform predictions on the list of images

    for r in image_results:
        print("r ", image_results.index(r))

        temp_list = r.probs.top5
        score_list = r.probs.top5conf

        # print("score list", score_list)
        imageList = r.names
        # print(imageList[temp_list[0]])
        for i in range(5):
            num = (image_results.index(r))
            score = score_list[i].item()
            # more specific
            if score >= 0.95:
                name = imageList[temp_list[i]]
                # less Specific
                tempName = imageList[temp_list[i]]
                tempName = tempName.replace("_", " ")
                # name and the one hot encoding val
                temp = similarWords3(w2v, tempName)
                # print("temp", temp)
                results.append(temp)

    del modelYOLO
    # print(results)
    df = pd.DataFrame(results, columns=['Name', 'Class'])
    print("df\n", df)
    df.to_excel("output.xlsx")

    return df


def addROItoDF(args, test_img_dir, test_img_list, lh_fmri, rh_fmri, ImgClasses, length):
    data = []
    length = len(ImgClasses)
    listroi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA",
               "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
               "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral",
               "parietal"]

    for img in range(length):
        excelData = []
        index = ImgClasses['Class'].loc[ImgClasses.index[img]]

        hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
        for roi in listroi:
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
                    lh_fmri[index, np.where(challenge_roi)[0]]
            elif hemisphere == 'right':
                fsaverage_response[np.where(fsaverage_roi)[0]] = \
                    rh_fmri[index, np.where(challenge_roi)[0]]
            accuracy = np.mean(fsaverage_response[np.where(fsaverage_roi)[0]])

            excelData.append(accuracy)
        data.append(excelData)

    columns = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA",
               "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
               "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral",
               "parietal"]

    df1 = pd.DataFrame(data, columns=columns)
    df = pd.concat([ImgClasses, df1], axis=1)
    print(df)
    df.to_excel("output2.xlsx")

    # Fill NaN values (where there is no accuracy score) with zeros
    df = df.fillna(0)
    df.to_excel('class_data.xlsx', index=False)

    X = df['Class'].values.reshape(-1, 1)
    y = df.drop(['Class', 'Name'], axis=1)

    print('X', X)
    print('Y', y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.to('cuda')
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    linear_mse = mean_squared_error(y_test, linear_predictions)
    print(f'Linear Regression Mean Squared Error: {linear_mse}')
    accuracy_score = linear_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)

    # Random Forest Regression (as previously provided)
    random_forest_model = RandomForestRegressor()
    random_forest_model.to('cuda')
    random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)
    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')
    accuracy_score = random_forest_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)

    return df


def makePred(df, val_list):
    X = df['Class'].values.reshape(-1, 1)
    y = df.drop(['Class', 'Name'], axis=1)

    # X2 =

    print('X', X)
    print('Y', y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_predictions)
    print(f'Linear Regression Mean Squared Error: {linear_mse}')
    accuracy_score = linear_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)

    # Random Forest Regression (as previously provided)
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)

    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')
    accuracy_score = random_forest_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)


# Main function used to classify the images into a category
def moreWords(results):
    category_counts = {}
    img_and_category = []
    category_names = []
    w2v = api.load("word2vec-google-news-300")

    for result in results:
        # print(result)
        probs = result.probs
        class_index = probs.top5
        listofsyn = []
        # clean data
        for class1 in class_index:
            newWord = result.names[class1].replace("_", " ")
            listofsyn.append(newWord)

        print('list', listofsyn)
        print(listofsyn)

        # finds top 5 words associated with the prediction
        category = similarWords3(w2v, listofsyn)
        # category = newWord
        # if there is a category, classify it to the categories we predefine for the model
        if category != "None":
            category = classifytoClasses(w2v, category)
            img_and_category.append(category)
        else:
            img_and_category.append("None")
        # count the different categories for analytics
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
            category_names.append(category)
    del w2v

    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # Create a bar chart, because I like charts -> Ali
    plt.figure(figsize=(12, 12))
    plt.barh(categories, counts, color='skyblue')

    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.title('Category Counts')
    plt.gca().invert_yaxis()  # Invert the y-axis to show the top category at the top
    plt.show()
    return img_and_category


def similarWords5(model, input_word):
    similar_words1 = []
    similar_words2 = []
    similar_words3 = []
    similar_words4 = []
    similar_words5 = []
    best_similarity_score = -1  # Initialize with a very low value

    # Get the list of English words from NLTK corpus
    english_words = set(words.words())
    # Get similar words to the input word
    try:
        similar_words1 = model.most_similar(input_word[0], topn=5)
    except KeyError as e:
        pass
    try:
        similar_words2 = model.most_similar(input_word[1], topn=5)
    except KeyError as e:
        pass
    try:
        similar_words3 = model.most_similar(input_word[2], topn=5)
    except KeyError as e:
        pass
    try:
        similar_words4 = model.most_similar(input_word[3], topn=5)
    except KeyError as e:
        pass
    try:
        similar_words5 = model.most_similar(input_word[4], topn=5)
    except KeyError as e:
        pass

    print(input_word[0], input_word[1], input_word[2], input_word[3], input_word[4])

    # Combine and filter similar words from both input words
    for word, similarity_score in similar_words1 + similar_words2 + similar_words3 + similar_words4 + similar_words5:

        # Check if the word is an English word and has a higher similarity score than the current best score
        if word in english_words and similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
            best_word = word

    # Check if a best word was found
    if best_similarity_score > -1:
        new_word = best_word
    else:
        new_word = "No suitable word found"
        return ['None', -1]

    # print(f"Input Words: {input_word}")
    # print(f"New Word: {new_word}")
    return new_word


def classifytoClasses(model, input_word):
    predefined_words = [
        "Mammals", "Birds", "Reptiles", "Amphibians", "Insects", "Aquatic", "Animals", "Wildlife", "Pets", "Farm",
        "Faces", "Emotions", "Body", "Clothing", "Person", "Fruits", "Vegetables", "Meat", "Dairy", "Grains",
        "Sweets", "Landscapes", "City", "Architecture", "Forests", "Mountains", "Beaches", "Deserts", "Oceans",
        "Rivers", "Lakes", "Sky", "Weather", "Night", "Planets", "Space", "Stars", "Trees", "Flowers", "Cars",
        "Bicycles", "Trains", "Aircraft", "Ships", "Vehicles", "Sports", "Music", "Art", "Books", "Movies",
        "Computers", "Smartphones", "Technology", "Industry", "Tools", "Fashion", "Jewelry", "Toys", "Games", "Money",
        "Buildings", "History", "Antiques", "Shapes", "Color", "Patterns", "Shadows", "Reflections", "Light", "Dark",
        "Textures", "Smoke", "Fire", "Snow", "Rain", "Lightning", "Disasters", "Vehicles",
        "Waterfalls", "Bridges", "Organisms", "Astronomy", "Water", "Rocks", "Seasons", "Photography", "Street",
        "Abstractions", "Illusions", "Silhouettes", "Objects", "Kitchen", "Games", "Room", "Athlete", "Bathroom",
        "Bedroom", "Plates",
        "Plants", "Instruments", "Food", "Drinks", "Toys", "Flags", "Cultures", "Religions", "Holidays", "Landmarks",
        "Transportation", "Cities", "Rural", "Mountains", "Islands", "Landscapes", "Weather", "Sunsets", "Dusk", "Dawn",
        "Abstract", "Minimalism", "Macro", "Micro", "Eco", "Vintage", "Retro", "Modern", "Traditional", "Sculpture",
        "Painting", "Dance", "Sculpture", "Sculpture", "Abstract", "Surrealism", "Impressionism", "Expressionism",
        "Cubism", "Realism", "Contemporary", "Minimalism", "Abstract", "Classical", "Experimental", "Folk", "Restaurant"
    ]
    predefined_words2 = [
        "Outdoor", "Food", "Indoor", "Appliance", "Sports", "Person", "Animal", "Vehicle", "Furniture", "Accessory",
        "Electric", "Kitchen"
    ]

    # Calculate word similarities
    similarities = {}
    threshold = -1
    returnval = "None"
    # Look through the predefine word to categorize the words
    for predefined_word in predefined_words:
        similarity = model.similarity(input_word, predefined_word)
        if similarity > threshold:
            returnval = predefined_word
            threshold = similarity

        similarities[predefined_word] = similarity
    index = predefined_words.index(returnval)
    # print([returnval, index])
    if returnval == "None":
        return ['None', -1]
    return [returnval, index]


def similarWords3(model, word):
    best_similarity_score = -1  # Initialize with a very low value

    # Get the list of English words from NLTK corpus
    english_words = set(words.words())
    try:
        if word.__contains__(' '):
            maybe = word.split()
            print(maybe)
            input_words0 = model.most_similar(maybe[0], topn=5)
            input_words1 = model.most_similar(maybe[1], topn=5)
            input_words = input_words0 + input_words1
        else:
            input_words = model.most_similar(word, topn=5)
        # print("input words", input_words)
    except KeyError:
        return ['None', -1]

    # Combine and filter similar words from all input words
    for word, similarity_score in input_words:
        #     Check if the word is an English word and has a higher similarity score than the current best score
        if word in english_words and similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
            best_word = word

    # Check if a best word was found
    if best_similarity_score > -1:
        new_word = best_word
    else:
        new_word = "No suitable word found"
        print(new_word)
        return ["None", -1]
    temp = classifytoClasses(model, new_word)
    return temp


def makeClassifications(df, img_list, img_dir):
    print("make classifications")
    w2v = api.load("word2vec-google-news-300")
    num_list = df['Num']
    train_img_list = makeList(img_dir, img_list, num_list)
    # print("train images\n", train_img_list)
    modelYOLO = YOLO('yolov8n-cls.pt')
    image_results = modelYOLO.predict(train_img_list, stream=True)
    #print(len(image_results))
    # print("num list\n", num_list)
    # print(train_img_list)

    results = []

    # Perform predictions on the list of images

    for r in image_results:
       #print(r)
        #print(image_results.index(r))

        temp_list = r.probs.top5
        score_list = r.probs.top5conf

        # print("score list", score_list)
        imageList = r.names
        # print(imageList[temp_list[0]])
        for i in range(5):

            #num = (image_results.index(r))
            score = score_list[i].item()
            # more specific
            if score >= 0.25:
                name = imageList[temp_list[i]]
                # less Specific
                tempName = imageList[temp_list[i]]
                tempName = tempName.replace("_", " ")
                # name and the one hot encoding val
                temp = similarWords3(w2v, tempName)
                print("temp", temp)
                results.append(temp)

    del modelYOLO
    print(results)
    df = pd.DataFrame(results, columns=['Name', 'Class'])
    print(df)
    return df


#def makeMorePred(lh_train, rh_train, lh_val, rh_val):
def makeMorePred(lh_train):
    # Random Forest Regression (as previously provided)
    random_forest_model = RandomForestRegressor()
    X = lh_train['Class']
    y = lh_train.drop(['Name', 'Class', 'Num'], axis=1)
    print('X', X)
    print('Y', y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regression (as previously provided)
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)

    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')
    accuracy_score = random_forest_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)
