# DSIM 2024


This repository contains the files for the course project.

## Intro

This project is composed of 3 parts:
* Voice emotion recognition with CNN model
* Panorama stiching for more than 2 images
* DCGAN training for cars images generation

Currently the UI of project can be reached by this [link](https://signal.trplai.com/). (works from 18:00)


Data for voice classification is taken from [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
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

The code is logically splitted for next subfolders:
### gan  ###

* `DSIM_gan.ipynb` - notebook with training process of gan, the training was run in the colab
* `ganfunc.py` - functions to build the dash graphs
* `generator.pth` - weights for torch generator

### panorama  ###

The folder contains fucntions and utilities for panorama in UI:

* `panorama.py` - functions to build panorama.
* `plots.py` - functions to generate plots

### voice  ###
* `cnn_model.joblib` - saved model for voice classification
* `data_augmentation.py` - move files to folders by the tone of voice
* `cnn_training.py` - training of the classification model
* `data_retrieving.py` - get data from kaggle
* `feature_extraction.py` - extract features from audio data




