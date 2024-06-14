import cv2
import numpy as np
import matplotlib.pyplot as pls
import pandas as pd

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)
    return image, gray, blurred, edged

def detect_contours(edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

# Example usage
image_path = 'data/test.jpeg'  # Replace with the path to your image
image, gray, blurred, edged = preprocess_image(image_path)
contours = detect_contours(edged)
contour_image = draw_contours(image, contours)

# Show or save the contour image
cv2.imwrite('contours.jpg', contour_image)



def save_cropped_coins(image, contours, output_dir='cropped_coins'):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_coin = image[y:y+h, x:x+w]
        cv2.imwrite(f'{output_dir}/coin_{i}.png', cropped_coin)

# Example usage
save_cropped_coins(image, contours)



path = 'data/test.jpeg'

image, edged = preprocess_image("data/test.jpeg")
contours = detect_contours(edged)
result_image = classify_coins(image.copy(), contours)
cv2.imshow("Classified Coins", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def show_image_with_matplotlib(image, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image_with_matplotlib(result_image)

if __name__ == "__main__":
    image_path = 'data/test.jpeg'
    image, edged = preprocess_image(image_path)
    contours = detect_contours(edged)
    result_image = classify_coins(image.copy(), contours)
    
    cv2.imshow("Classified Coins", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


coins = cv2.imread('data/coins_noize_1.jpg')
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9,9), 2)




circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=100, param2=30,
                               minRadius=25, maxRadius=50)

coins_setected = len(circles[0])

plot_HoughCircles(circles[0].astype(int), coins)



#
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
    minRadius = trial.suggest_int('minRadius', 50, 150)
    maxRadius = trial.suggest_int('maxRadius', 0, 60)
    
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