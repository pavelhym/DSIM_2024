import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os





def plot_HoughCircles(circles, img, eps=5):
    mask = np.zeros_like(img)
    for circle in circles:
        x, y, R = circle

        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                if ((i ** 2 + j ** 2) > (R - eps // 2) ** 2) & ((i ** 2 + j ** 2) < (R + eps // 2) ** 2):
                    mask[np.clip(y + i, 0, img.shape[0] - 1), 
                         np.clip(x + j, 0, img.shape[1] - 1)] = [0, 0, 255]
        
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(np.maximum(img, mask), cv2.COLOR_BGR2RGB) , cmap = 'gray')




### Example

coins = cv2.imread('data/coins_noize_1.jpg')
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9,9), 2)


circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=100, param2=30,
                               minRadius=25, maxRadius=50)

coins_setected = len(circles[0])
plot_HoughCircles(circles[0].astype(int), coins)







# Quality of number of found images
data = pd.read_csv("data/coins_count_values.csv")
data['folder'].unique()

data_euro = data[data['folder'] =='euro_coins']


error = []
for index, row in data_euro.iterrows():
    name = row['image_name']
    real_num = row['coins_count']

    coins = cv2.imread(f'data/euro_coins/{name}')
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)




    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                                param1=100, param2=30,
                                minRadius=25, maxRadius=50)
    try:
        coins_setected = len(circles[0])
    except:
        coins_setected = 0
    
    error.append(coins_setected - real_num)


np.mean(np.abs(error))




#Trying to find right parameters

import optuna

def coin_finder(param1 = 100,param2=30, minRadius=25, maxRadius=50):
    error = []
    for index, row in data_euro.iterrows():
        name = row['image_name']
        real_num = row['coins_count']

        coins = cv2.imread(f'data/euro_coins/{name}')
        gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)




        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                                    param1=param1, param2=param2,
                                    minRadius=minRadius, maxRadius=maxRadius)
        try:
            coins_setected = len(circles[0])
        except:
            coins_setected = 0
        
        error.append(coins_setected - real_num)

    return np.mean(np.abs(error))
    


def objective(trial):
    # Suggest values for the hyperparameters
    param1 = trial.suggest_int('param1', 20, 150)
    param2 = trial.suggest_int('param2', 10, 50)
    minRadius = trial.suggest_int('minRadius', 50, 100)
    maxRadius = trial.suggest_int('maxRadius', 60, 150)

    if maxRadius <= minRadius:
        raise optuna.exceptions.TrialPruned()
    
    # Compute the value of the custom function
    return coin_finder(param1, param2, minRadius, maxRadius)



# Create a study object
study = optuna.create_study(direction='minimize')

# Optimize the objective function
study.optimize(objective, n_trials=100)

# Print the best parameters and value
print('Best parameters:', study.best_params)
print('Best value:', study.best_value)


study.best_params




##Check with best parameters
error = []
for index, row in data_euro.iterrows():
    name = row['image_name']
    real_num = row['coins_count']

    coins = cv2.imread(f'data/euro_coins/{name}')
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)




    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                                param1=99, param2=40,
                                minRadius=14, maxRadius=53)
    try:
        coins_setected = len(circles[0])
    except:
        coins_setected = 0
    
    try:
        plot_HoughCircles(circles[0].astype(int), coins)
    except:
        continue
    
    error.append(coins_setected - real_num)


np.mean(np.abs(error))









### Example





def return_detections_manual(path, show = False, return_circle = False):
    detections = []
    coins = cv2.imread(path)
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)


    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                                param1=99, param2=40,
                                minRadius=14, maxRadius=35)


    if show:
        plot_HoughCircles(circles[0].astype(int), coins)

    if return_circle:
        return circles

    try:
        for (x, y, r) in circles[0]:
                mask = np.zeros_like(gray) #Create a mask of the same dimension as the original


                center = (int(x), int(y))
                radius = int(r) #Identify center and radius of the circle
                cv2.circle(mask, center, radius, (255, 255, 255), -1) #White circle


                masked_data = cv2.bitwise_and(gray, gray, mask=mask) # Apply the mask

                # Cut the coin
                x1 = max(0, center[0] - radius)
                y1 = max(0, center[1] - radius)
                x2 = min(coins.shape[1], center[0] + radius)
                y2 = min(coins.shape[0], center[1] + radius)
                cropped_coin = masked_data[y1:y2, x1:x2]
                detections.append((1.0, x1, y1,x2 , y2))
        return detections
    except:
        return detections




def convert_yolo_to_pixel(yolo_box, img_width, img_height):
    class_id, x_center, y_center, width, height = yolo_box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    
    return class_id, x_min, y_min, x_max, y_max



def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area



def match_boxes(detections, ground_truths, iou_threshold=0.5):

    tp = 0
    fp = 0
    matched_gt = set()

    for det in detections:
        _, det_xmin, det_ymin, det_xmax, det_ymax = det
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            _, gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt
            if i in matched_gt:
                continue
            iou = calculate_iou((det_xmin, det_ymin, det_xmax, det_ymax), (gt_xmin, gt_ymin, gt_xmax, gt_ymax))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou > iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)



# Assume detections and ground truths are lists of boxes in the same format
# Detections: [(class, xmin, ymin, xmax, ymax), ...]
# Ground truths: [(class, x_center, y_center, width, height), ...]


def real_labels_parsing(path):

    img_width = 512
    img_height = 512

    # Parse ground truth boxes
    ground_truths = []
    with open(path, 'r') as f:
        for line in f.readlines():
            yolo_box = list(map(float, line.strip().split()))
            gt_box = convert_yolo_to_pixel(yolo_box, img_width, img_height)
            ground_truths.append(gt_box)
    return ground_truths



all_names = [x[:-4] for x in os.listdir("Small-object-detection-for-euro-coins-5/test/images")]


total_tp = 0
total_fp = 0
total_fn = 0


for name in all_names:
    image_path = f'Small-object-detection-for-euro-coins-5/test/images/{name}.jpg'
    label_path = f'Small-object-detection-for-euro-coins-5/test/labels/{name}.txt'

    detections = return_detections_manual(image_path)
    ground_truths = real_labels_parsing(label_path)
    if len(ground_truths) < 1:
        continue

    tp, fp, fn = match_boxes(detections, ground_truths, iou_threshold=0.6)

    total_tp += tp
    total_fp += fp
    total_fn += fn
# Calculate precision
precision = calculate_precision(tp, fp)
print(f'Precision: {precision}')



detections = return_detections_manual(image_path, show = True)
ground_truths = real_labels_parsing(label_path)




import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_yolo_labels(file_path, img_width, img_height):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            yolo_box = list(map(float, line.strip().split()))
            class_id, x_center, y_center, width, height = yolo_box
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)
            
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes

def plot_ground_truth_and_detections(image_path, gt_boxes, detections, img_width, img_height):
    # Load image
    img = Image.open(image_path)
    
    # Create plot
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Plot ground truth boxes
    for gt in gt_boxes:
        class_id, x_min, y_min, x_max, y_max = gt
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth')
        ax.add_patch(rect)
    
    # Plot detected circles
    for det in detections:
        det_x_center, det_y_center, det_radius = det
        circle = patches.Circle((det_x_center, det_y_center), det_radius, linewidth=2, edgecolor='r', facecolor='none', label='Detection')
        print("added")
        ax.add_patch(circle)
    
    # Add legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()





for name in all_names:
    image_path = f'Small-object-detection-for-euro-coins-5/test/images/{name}.jpg'
    label_path = f'Small-object-detection-for-euro-coins-5/test/labels/{name}.txt'



    try:
        detections = return_detections_manual(image_path, show=False, return_circle=True)[0]
        ground_truths = real_labels_parsing(label_path)
    except:
        continue

    if len(detections) > 0 and len(ground_truths) > 0:
        plot_ground_truth_and_detections(image_path, ground_truths, detections, 512, 512)






#Get whole model together


import xgboost as xgb

model_xgb = xgb.XGBClassifier()
model_xgb.load_model('weights/xgboost_model.xgb')

import pickle
loaded_label_encoder = pickle.load(open("weights/label_encoder.pkl", 'rb'))





def return_detections_manual_with_pred(path, show = False, return_circle = False):
    detections = []
    coins = cv2.imread(path)
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)


    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                                param1=99, param2=40,
                                minRadius=14, maxRadius=35)


    if show:
        plot_HoughCircles(circles[0].astype(int), coins)

    if return_circle:
        return circles

    try:
        for (x, y, r) in circles[0]:
                mask = np.zeros_like(gray) #Create a mask of the same dimension as the original


                center = (int(x), int(y))
                radius = int(r) #Identify center and radius of the circle
                cv2.circle(mask, center, radius, (255, 255, 255), -1) #White circle


                masked_data = cv2.bitwise_and(gray, gray, mask=mask) # Apply the mask

                # Cut the coin
                x1 = max(0, center[0] - radius)
                y1 = max(0, center[1] - radius)
                x2 = min(coins.shape[1], center[0] + radius)
                y2 = min(coins.shape[0], center[1] + radius)
                cropped_coin = masked_data[y1:y2, x1:x2]


                cropped_coin = cv2.resize(cropped_coin, (64,64)) # Resize the image
                coin_flatten = np.expand_dims(cropped_coin.flatten(), axis=0)

                coin_stand = (coin_flatten - np.mean(coin_flatten))/np.std(coin_flatten)

                det_class = model_xgb.predict(coin_flatten)
                det_class_trans = loaded_label_encoder.inverse_transform(det_class)[0]
                detections.append((det_class_trans, x1, y1,x2 , y2))
        return detections
    except:
        return detections
    


yolo_dict = {
"0" : "EUR-1-cent",
"1" : "EUR-1-euro",
"2" : "EUR-10-cent",
"3" : "EUR-2-cent",
"4" : "EUR-2-euro",
"5" : "EUR-20-cent",
"6" : "EUR-5-cent",
"7" : "EUR-50-cent",
}


xgb_dict = {
 "54" : "EUR-1-cent",
 "55" : "EUR-2-cent",
 "56" : "EUR-5-cent",
 "57" : "EUR-10-cent",
 "58" : "EUR-20-cent",
 "59" : "EUR-50-cent",
 "60" : "EUR-1-euro",
 "61" : "EUR-2-euro",
}



def plot_model_prediction(image_path, detections):
    image_path = image_path
    image = cv2.imread(image_path)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for det in detections:
        cl_det, x1, y1, x2, y2 = det 
        label = xgb_dict[cl_det] 
        # Draw the box and label on the image
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image_rgb, f'{label}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def plot_real_labels(image_path, ground_truths):
    image_path = image_path
    image = cv2.imread(image_path)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for det in ground_truths:
        cl_det, x1, y1, x2, y2 = det 
        label = yolo_dict[str(int(cl_det))] 
        # Draw the box and label on the image
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 0), 2)
        cv2.putText(image_rgb, f'{label}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()





for name in all_names:
    image_path = f'Small-object-detection-for-euro-coins-5/test/images/{name}.jpg'
    label_path = f'Small-object-detection-for-euro-coins-5/test/labels/{name}.txt'


    try:
        detections = return_detections_manual_with_pred(image_path, show=False)
        ground_truths = real_labels_parsing(label_path)
    except:
        continue

    if len(detections) > 0 and len(ground_truths) > 0:
        print(image_path)
        plot_model_prediction(image_path, detections)
        plot_real_labels(image_path, ground_truths)



