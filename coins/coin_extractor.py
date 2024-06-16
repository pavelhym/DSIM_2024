import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






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