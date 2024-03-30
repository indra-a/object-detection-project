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
|   train.py
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
|   |           
|   +---models
|   |   |   model_architecture.py
|   |   |   optimizer.py
|   |   |   
|   |           
|   +---utils
|   |   |   data_utils.py
|   |   |   metrics_utils.py
|   |   |   mlflow_utils.py
|   |   |   __init__.py
|   |   |   
|   |           
|           
\---tests
        __init__.py
        

```

<b>Setup</b>

1. Clone repository 

```
git clone https://github.com/indra-a/object-detection-project.git
cd object-detection-project
```

2. Install packages 

```
pip install -r requirement.txt
```

3. Download data and make changes to ```config.yaml```

```config.yaml
data:
  label_path: # Enter path to labels xlsx
  images_path: # Enter path to directory of images

preprocess:
  resize: [224, 224]

model:
  name: lenet
  channels: 32
  kernel_size: 5
  stride: 1
  input_size: 224
  output_size: 2
  dropout: 0.0
  pretrained: false

...
```

<b>Training model</b>

```
python train.py
```

<b>Deployment</b>

After training, you can deploy the model using the app/app.py script. This script loads the trained model from MLflow and creates a Streamlit application for serving the model.

To run the deployment app, execute the following command:
```
streamlit run app/app.py
```
This will start the Streamlit server and open the application in your default web browser, where you can upload images and obtain predictions from the deployed model.