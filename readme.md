# DSIM 2024


This repository contains the files for the course project.

Currently the UI of project can be reached by this [link](https://signal.trplai.com/)

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

The structure of project folders is:


## gan  ##

* `DSIM_gan.ipynb` - notebook with training process of gan, the training was run in the colab
* `ganfunc.py` - functions to build the dash graphs
* `generator.pth` - weights for torch generator

## panorama  ##

The folder contains fucntions and utilities for panorama in UI:

* `panorama.py` - functions to build panorama.
* `plots.py` - functions to generate plots

## voice  ##
* `cnn_model.joblib` - saved model for voice classification
* `audios` - couples of samples used for the training of the model (entire dataset can be seen at the following [link](https://drive.google.com/drive/folders/1NWc7uNXmKP--r3JiFZf5HRuQ43qCcmir?usp=drive_link))
* `cnn_training.py` - code for the training process of the model, the training was run in the colab
* `data_retrieving.py` -  code for the retrieving of the audios from Kaggle
* `data_augmentation.py` - code for the augmentation of the audio files (augmented dataset can be found at the following [link](https://drive.google.com/drive/folders/1-9gY4ZMZZnXHPJclTyR50QIO9OkGK4n_?usp=drive_link))
* `feature_extraction.py` - code for extracting the features from the audio files (features can be found at the following [link](https://drive.google.com/drive/folders/1v4DhlAleA2eWBVut-6d_QKn7t-EKBkQg?usp=drive_link))



