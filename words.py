import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO


def Predictions(train, train_fmri, val, val_fmri):

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(train, train_fmri)
    linear_regression_predictions = linear_regression_model.predict(val)

    #print(val_fmri, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", linear_regression_predictions)
    #linear_regression_mse = mean_squared_error(val_fmri, linear_regression_predictions)
    #print(f'Random Forest Mean Squared Error: {linear_regression_mse}')
    #score = linear_regression_model.score(val, val_fmri)
    #print("accuracy score", score)

    return linear_regression_predictions


def make_classifications(image_list, idxs, device, batch_size=100):
    modelYOLO = YOLO('yolov8n.pt')
    modelYOLO.to(device)
    results = []

    words = ['person', 'animal', 'vehicle', 'outdoor', 'accessory', 'toy','container', 'utensil', 'food', 'furniture', 'appliance','indoor', 'clothing'] 

    for start_idx in range(0, len(image_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = image_list[start_idx:end_idx]
        batch_idxs = idxs[start_idx:end_idx]

        image_results = modelYOLO.predict(batch_imgs, stream=True)
        names = modelYOLO.names

        for i, result in enumerate(image_results):
            detection_count = result.boxes.shape[0]
            image_idx = batch_idxs[i]

            best_confidence = 0.3
            best_item = None
            for j in range(len(result.boxes)):
                confidence = float(result.boxes.conf[j].item())
                if confidence > best_confidence:
                    cls = int(result.boxes.cls[j].item())
                    name = names[cls]
                    name = class_mapping.get(name)
                    num = words.index(name)
                    best_confidence = confidence

                    if best_item is None or confidence > best_item['confidence']:
                        best_item = {
                            'cls': cls,
                            'name': name,
                            'num': num,
                            'confidence': confidence
                        }
            if best_item:
                results.append([cls, num])
            else:

                results.append([-1, -1])

    df = pd.DataFrame(results)
    df = df.fillna(-1)
    final = df.to_numpy()

    return final


class_mapping2 = {
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
    'suitcase': 'luggage',
    'vase': 'home',
    'skis': 'sports',
    'hot dog': 'food',
    'frisbee': 'toy',
    'backpack': 'luggage',
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

class_mapping = { 
    'person': 'person', 
    'bicycle': 'vehicle', 
    'car': 'vehicle', 
    'motorcycle': 'vehicle', 
    'airplane': 'vehicle', 
    'bus': 'vehicle', 
    'train': 'vehicle', 
    'truck': 'vehicle', 
    'boat': 'vehicle', 
    'traffic light': 'outdoor', 
    'fire hydrant': 'outdoor', 
    'stop sign': 'outdoor', 
    'parking meter': 'outdoor', 
    'bench': 'outdoor', 
    'bird': 'animal', 
    'cat': 'animal', 
    'dog': 'animal', 
    'horse': 'animal', 
    'sheep': 'animal', 
    'cow': 'animal', 
    'elephant': 'animal', 
    'bear': 'animal', 
    'zebra': 'animal', 
    'giraffe': 'animal', 
    'backpack': 'accessory', 
    'umbrella': 'accessory', 
    'handbag': 'accessory', 
    'tie': 'accessory', 
    'suitcase': 'accessory', 
    'frisbee': 'toy', 
    'skis': 'outdoor', 
    'snowboard': 'outdoor', 
    'sports ball': 'toy', 
    'kite': 'toy', 
    'baseball bat': 'outdoor',
    'baseball glove': 'outdoor', 
    'skateboard': 'outdoor', 
    'surfboard': 'outdoor', 
    'tennis racket': 'outdoor', 
    'bottle': 'container', 
    'wine glass': 'container', 
    'cup': 'container', 
    'fork': 'utensil', 
    'knife': 'utensil', 
    'spoon': 'utensil', 
    'bowl': 'container', 
    'banana': 'food', 
    'apple': 'food', 
    'sandwich': 'food', 
    'orange': 'food', 
    'broccoli': 'food', 
    'carrot': 'food', 
    'hot dog': 'food', 
    'pizza': 'food', 
    'donut': 'food', 
    'cake': 'food', 
    'chair': 'furniture', 
    'couch': 'furniture', 
    'potted plant': 'furniture', 
    'bed': 'furniture', 
    'dining table': 'furniture', 
    'toilet': 'furniture', 
    'tv': 'appliance', 
    'laptop': 'appliance', 
    'mouse': 'appliance', 
    'remote': 'appliance', 
    'keyboard': 'appliance', 
    'cell phone': 'appliance', 
    'microwave': 'appliance', 
    'oven': 'appliance', 
    'toaster': 'appliance', 
    'sink': 'appliance', 
    'refrigerator': 'appliance', 
    'book': 'indoor', 
    'clock': 'indoor', 
    'vase': 'indoor', 
    'scissors': 'indoor', 
    'teddy bear': 'toy', 
    'hair drier': 'appliance', 
    'toothbrush': 'indoor' 

} 