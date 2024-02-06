import os

import gensim.downloader as api
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from nltk.corpus import words
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
            input_words = []
            for i in range(len(maybe)):
                # print(maybe)
                input_words0 = model.most_similar(maybe[0], topn=5)
                input_words = input_words + input_words0

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
        # print(new_word)
        return ["None", -1]
    temp = classifytoClasses(model, new_word)
    return temp



def makeMorePred(train, val):
    # Random Forest Regression (as previously provided)
    # random_forest_model = RandomForestRegressor()
    X_train = train['Class'].to_numpy().reshape(-1, 1)
    y_train = train.drop(['Class'], axis=1).to_numpy()  # .reshape(-1, 1)
    X_test = val['Class'].to_numpy().reshape(-1, 1)
    y_test = val.drop(['Class'], axis=1).to_numpy()  # .reshape(-1, 1)
    print(len(X_train))
    print(len(y_train))
    print(y_train)

    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train, y_train)

    random_forest_predictions = random_forest_model.predict(X_test)

    print(y_test, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", random_forest_predictions)
    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')
    accuracy_score = random_forest_model.score(X_test, y_test)
    # print("accuracy score", accuracy_score)
    print("accuracy score", accuracy_score)
    return random_forest_predictions


def makeClassifications(idxs, img_list, img_dir, batch_size=1000):
    w2v = api.load("word2vec-google-news-300")
    train_img_list = makeList(img_dir, img_list, idxs)
    modelYOLO = YOLO('yolov8n.pt')

    results = []

    for start_idx in range(0, len(train_img_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = train_img_list[start_idx:end_idx]

        # Perform predictions on the batch of images
        image_results = modelYOLO.predict(batch_imgs, stream=True)
        # data = []

        for r in image_results:
            temp = set()  # Use a set to store unique items
            detection_count = r.boxes.shape[0]

            for i in range(min(detection_count, 5)):  # Limit to a maximum of 5 items
                cls = int(r.boxes.cls[i].item())
                name = r.names[cls]
                confidence = float(r.boxes.conf[i].item())
                bounding_box = r.boxes.xyxy[i].cpu().numpy()
                x = int(bounding_box[0])
                y = int(bounding_box[1])
                width = int(bounding_box[2] - x)
                height = int(bounding_box[3] - y)
                temp.add(cls)  # Add class to the set

            results.append(list(temp))

    del modelYOLO
    # df = pd.DataFrame(data)  # , columns=['Num','Name', 'Class'])
    Data_type = object
    # data = np.array(results, dtype=Data_type)

    print(results)
    data = pd.DataFrame(results)
    data.fillna(-1, inplace=True)
    # data = results.reshape(-1, 1)
    print(data)
    return data
