# DSIM 2024


This repository contains the files for the course project.

## Intro

This project is composed of 4 parts:
* Voice emotion recognition with CNN model
* Panorama stiching for more than 2 images
* Coin detection and classification using two approaches
* DCGAN training for cars images generation

Currently the UI of project can be reached by this [link](https://signal.trplai.com/). (works from 18:00)


Data for voice classification is taken from [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
Data for coin detection and classification is taken from [here](https://universe.roboflow.com/hlcv2023finalproject/small-object-detection-for-euro-coins/dataset/5) and [here](https://www.kaggle.com/datasets/wanderdust/coin-images).
Data of car images for DCGAN is taken from [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).

## Installation
You can build docker image from by:

```bash
sudo docker build -t signalui .
```

And then run a contaner by:

```bash
sudo docker run -it -p 8506:8506 --name signal --restart unless-stopped -d  signalui
```

Or alternatively you can run streamlit UI from your command promt with all requirements installed by:

```bash
streamlit run dashboard.py
```

## Repo structure
Main folder contains these files:
* `dashboard.py` - python code for streamlit application
* `Dockerfile` - dockerfile for image building

The code is logically splitted into next subfolders:
### gan  ###

* `DSIM_gan.ipynb` - notebook with training process of gan, the training was run in the colab
* `ganfunc.py` - functions to build the dash graphs
* `generator.pth` - weights for torch generator

### panorama/src  ###

The folder contains functions and utilities for panorama in UI:

* `panorama.py` - functions to build panorama.
* `plots.py` - functions to generate plots
* `test.ipynb` - jupyter notebook to test panorama creation

### coins ###

This folder contains data, weights and code for the extraction and classification of coins. Specifically:
* **data:** contains images of coins
* **weights:** contains the best parameters of the model
* `Inference_Yolo.ipynb` code for the inference of the coins using deep model
* `YOLO_training+inference.ipynb` contains code for the training of the deep model
* `coin_classification_manual.ipynb` contains code for coin detection and classification using a manual approach
* `coin_extractor.py` code for the detection of coins
* `coinfunc.py` useful function for coin detection on the dashboard

### voice  ###
* `cnn_model.joblib` - saved model for voice classification
* `data_augmentation.py` - move files to folders by the tone of voice
* `cnn_training.py` - training of the classification model
* `data_retrieving.py` - get data from kaggle
* `feature_extraction.py` - extract features from audio data


## Demonstration  


![til](https://github.com/pavelhym/DSIM_2024/blob/main/demonstration.gif)


