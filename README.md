# Single Object Detection for Fovea localization in fundus images of AMD patients

In this project, my goal is to experiment with different Computer Vision model architectures and parameters in context of Single Object Detection problem. Starting with simple architecture and no additional regularization all the way to complex structures and regularizations, I built different models and compared their performance. The purpose of this project is to experiment with different combinations of model structure and model parameters.

<b>Problem Statement and Dataset</b>

In this dataset we are given fundus images of Age-related Macular degeneration (AMD) patients. The dataset has been release as a part of a challenge "Automatic Detection challenge on Age-related Macular degeneration" information and dataset of which is available here https://amd.grand-challenge.org/
Essentially, there are 4 tasks that make up the challenge, but I will only focus on Single Object Detection in this project.
In my different much simpler project, I addressed the image classification problem of this challenge.

<b>Directory Structure</b>

```markdown

|   config.yaml
|   README.md
|   requirements.txt
|   test.ipynb
|   text.txt
|   train.py
|   tree.md
|   
+---app
|       app.py
|       utils.py
|       __init__.py
|       
+---mlruns
|           
+---src
|   |   __init__.py
|   |   
|   +---data
|   |   |   dataset.py
|   |   |   data_augmentation.py
|   |   |   data_preprocessing.py
|   |   |   __init__.py
|   |   |   
|   |   \---__pycache__
|   |           dataset.cpython-310.pyc
|   |           data_augmentation.cpython-310.pyc
|   |           data_preprocessing.cpython-310.pyc
|   |           __init__.cpython-310.pyc
|   |           
|   +---models
|   |   |   model_architecture.py
|   |   |   optimizer.py
|   |   |   
|   |   \---__pycache__
|   |           model_architecture.cpython-310.pyc
|   |           optimizer.cpython-310.pyc
|   |           
|   +---utils
|   |   |   data_utils.py
|   |   |   metrics_utils.py
|   |   |   mlflow_utils.py
|   |   |   __init__.py
|   |   |   
|   |   \---__pycache__
|   |           mlflow_utils.cpython-310.pyc
|   |           __init__.cpython-310.pyc
|   |           
|   \---__pycache__
|           __init__.cpython-310.pyc
|           
\---tests
        __init__.py
        

```

<b>Setup</b>

1. Clone repository 

```git clone https://github.com/indra-a/object-detection-project.git```