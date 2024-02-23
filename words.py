import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from ultralytics import YOLO


def Predictions(train, train_fmri, val):
    print("PREDICTIONS")
    train = train.to_numpy()
    train_fmri = train_fmri.to_numpy()
    val = val.to_numpy()
   

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(train, train_fmri)
    linear_regression_predictions = linear_regression_model.predict(val)

    return linear_regression_predictions


def make_classifications(image_list, idxs, device, batch_size=64):
    modelYOLO = YOLO('yolov8n.pt')
    modelYOLO.to(device)
    results = []

    words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
             'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing',
               'computer', 'fruit', 'vegetable', 'tool']

    for start_idx in range(0, len(image_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = image_list[start_idx:end_idx]
        batch_idxs = idxs[start_idx:end_idx]

        image_results = modelYOLO.predict(batch_imgs, stream=True)
        names = modelYOLO.names
        # print(names)

        for i, result in enumerate(image_results):
            # print(i,result)
            detection_count = result.boxes.shape[0]
            image_idx = batch_idxs[i]

            best_confidence = 0.75
            # best_item = None
            for j in range(len(result.boxes)):
                confidence = float(result.boxes.conf[j].item())
                if confidence > best_confidence:
                    cls = int(result.boxes.cls[j].item())
                    name = names[cls]
                    name = class_mapping.get(name)
                    num = words.index(name)
                    # best_confidence = confidence
                    results.append([i, cls, num])
                    # print(i,cls,num)

                    #if best_item is None or confidence > best_item['confidence']:
                    #    best_item = {
                    #        'cls': cls,
                    #        'name': name,
                    #        'num': num,
                    #        'confidence': confidence
                    #    }
            #if best_item:
            #    results.append([cls, num])
            #else:
            #    results.append([-1, -1])

    df = pd.DataFrame(results)
    df = df.fillna(-1)
    print(df)
    final = df.to_numpy()

    return final


class_mapping = {
    'chair': 'furniture',
    'bowl': 'kitchenware',
    'dining table': 'furniture',
    'person': 'person',
    'bird': 'animal',
    'knife': 'kitchenware',
    'sink': 'appliance',
    'horse': 'animal',
    'cake': 'food',
    'giraffe': 'animal',
    'car': 'vehicle',
    'umbrella': 'accessory',
    'refrigerator': 'appliance',
    'cow': 'animal',
    'dog': 'animal',
    'tv': 'electronics',
    'surfboard': 'sports',
    'cat': 'animal',
    'stop sign': 'traffic',
    'train': 'vehicle',
    'zebra': 'animal',
    'carrot': 'vegetable',
    'bicycle': 'vehicle',
    'sports ball': 'sports',
    'sheep': 'animal',
    'motorcycle': 'vehicle',
    'bottle': 'kitchenware',
    'sandwich': 'food',
    'clock': 'home',
    'bear': 'animal',
    'truck': 'vehicle',
    'traffic light': 'traffic',
    'cell phone': 'electronics',
    'oven': 'appliance',
    'cup': 'kitchenware',
    'couch': 'furniture',
    'airplane': 'vehicle',
    'boat': 'vehicle',
    'bus': 'vehicle',
    'broccoli': 'vegetable',
    'tennis racket': 'sports',
    'elephant': 'animal',
    'parking meter': 'traffic',
    'tie': 'clothing',
    'bed': 'furniture',
    'toaster': 'appliance',
    'banana': 'fruit',
    'toothbrush': 'hygiene',
    'kite': 'toy',
    'skateboard': 'sports',
    'potted plant': 'home',
    'bench': 'outdoor',
    'donut': 'food',
    'spoon': 'kitchenware',
    'toilet': 'plumbing',
    'baseball bat': 'sports',
    'pizza': 'food',
    'scissors': 'tool',
    'fire hydrant': 'outdoor',
    'teddy bear': 'toy',
    'remote': 'electronics',
    'apple': 'fruit',
    'suitcase': 'accessory',
    'vase': 'home',
    'skis': 'sports',
    'hot dog': 'food',
    'frisbee': 'toy',
    'backpack': 'accessory',
    'microwave': 'appliance',
    'wine glass': 'kitchenware',
    'snowboard': 'sports',
    'baseball glove': 'sports',
    'book': 'toy',
    'orange': 'fruit',
    'fork': 'kitchenware',
    'laptop': 'electronics',
    'handbag': 'accessory',
    'keyboard': 'computer',
    'mouse': 'computer'
}
