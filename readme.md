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
* `audios` - couples of samples used for training the model (entire dataset can be seen at the following [link](https://drive.google.com/drive/folders/1NWc7uNXmKP--r3JiFZf5HRuQ43qCcmir?usp=drive_link)



