import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO
from data import makeList


def makePredictions(train, train_fmri, val, val_fmri):

    model = LinearRegression()
    model.fit(train, train_fmri)
    random_forest_predictions = model.predict(val)

    print(val_fmri, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", random_forest_predictions)
    random_forest_mse = mean_squared_error(val_fmri, random_forest_predictions)
    print(f'Mean Squared Error: {random_forest_mse}')
    accuracy_score = model.score(val, val_fmri)
    print("Accuracy Score", accuracy_score)


    return random_forest_predictions


def makeClassifications(idxs, img_list, img_dir, batch_size=500):
    # w2v = api.load("word2vec-google-news-300")
    torch.cuda.empty_cache()
    train_img_list = makeList(img_dir, img_list, idxs)
    modelYOLO = YOLO('yolov8n.pt')
    modelYOLO.to('cuda:1')

    results = []

    for start_idx in range(0, len(train_img_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = train_img_list[start_idx:end_idx]

        # Perform predictions on the batch of images
        image_results = modelYOLO.predict(batch_imgs, stream=True)
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
    print(results)
    data = pd.DataFrame(results)
    data.fillna(-1, inplace=True)
    print(data)
    return data
