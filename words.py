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
    w2v.to('cuda')
    num_list = df['Num']
    train_img_list = makeList(img_dir, img_list, num_list)
    # print("train images\n", train_img_list)
    modelYOLO = YOLO('yolov8n-cls.pt')

    image_results = modelYOLO.predict(train_img_list, stream=True)
    modelYOLO.to('cuda')
    # print(len(image_results))
    # print("num list\n", num_list)
    # print(train_img_list)

    results = []

    # Perform predictions on the list of images

    for r in image_results:

        temp_list = r.probs.top5
        score_list = r.probs.top5conf

        # print("score list", score_list)
        imageList = r.names
        # print(imageList[temp_list[0]])
        for i in range(5):

            # num = (image_results.index(r))
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


# def makeMorePred(lh_train, rh_train, lh_val, rh_val):
def makeMorePred(X_train, X_test, y_train, y_test):
    # Random Forest Regression (as previously provided)
    # random_forest_model = RandomForestRegressor()

    print('X\n', X_train, X_test)
    y_train = y_train.drop(['Class', 'Num', 'Name'], axis=1)
    y_test = y_test.drop(['Class', 'Num'], axis=1)
    # y = y.drop(['Class', 'Num'], axis=1)
    # print('Y', y)
    print('Y\n', y_train, y_test)

    # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regression (as previously provided)
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train, y_train)
    print(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)
    print(y_test, random_forest_predictions)
    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')
    accuracy_score = random_forest_model.score(X_test, y_test)
    print("accuracy score", accuracy_score)
