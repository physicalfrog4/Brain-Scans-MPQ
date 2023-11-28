import gensim.downloader as api
from matplotlib import pyplot as plt
from nltk.corpus import words
from ultralytics import YOLO


def wordClassifier(train_img_list):
    # Uncomment this if you want the word classifier
    modelYOLO = YOLO('yolov8n-cls.pt')
    # predicts what the image is based on the preloaded YOLO model.
    image_results = modelYOLO.predict(train_img_list)
    del modelYOLO
    # take the predictions and categorizes them
    ImgClasses = moreWords(image_results)
    return ImgClasses


# Main function used to classify the images into a category
def moreWords(results):
    category_counts = {}
    img_and_category = []
    category_names = []
    w2v = api.load("word2vec-google-news-300")

    for result in results:
        probs = result.probs
        class_index = probs.top5
        listofsyn = []
        # clean data
        for class1 in class_index:
            newWord = result.names[class1].replace("_", " ")
            listofsyn.append(newWord)
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
        return "None"

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
    return returnval


def similarWords3(model, input_words):
    similar_words = []
    best_similarity_score = -1  # Initialize with a very low value

    # Get the list of English words from NLTK corpus
    english_words = set(words.words())

    # Get similar words to each input word
    for word in input_words:
        try:
            similar_words.extend(model.most_similar(word, topn=5))
        except KeyError as e:
            pass

    # Combine and filter similar words from all input words
    for word, similarity_score in similar_words:
        # Check if the word is an English word and has a higher similarity score than the current best score
        if word in english_words and similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
            best_word = word

    # Check if a best word was found
    if best_similarity_score > -1:
        new_word = best_word
    else:
        new_word = "No suitable word found"
        return "None"

    # print(f"Input Words: {input_words}")
    # print(f"New Word: {new_word}")
    return new_word
