# Cell extraction in microscope image and deployment with Azure ML service

In this repo, we'll cover two topics, one is how to use Unet and Watershed algorithm to extract and segment cell from chemical material or biological tissue microscope image, the other is to use Azure Machine Learning Service to train and deploy the Unet model for productivity.

# Cell extraction in chemical or biological microscope image

## Introduction

In many industries, scientists always use microscope images to study the micro structure of some biological tissue or materials. They also need to seperate the cell from the complete structure, so that they can get the count of the cell and also the centroid and the area of the cell, to check the quality and metric of the tissue or materials. Microscope images are like below, left is polyurethane material and right is cell.

<p align="center"><img src="unet+watershed_research_code/image/polyurethane.jpg" width="30%" height="30%"> <img src="unet+watershed_research_code/image/cell.jpg" width="28%" height="28%">

This repo is to use [Unet deep learning network](https://en.wikipedia.org/wiki/U-Net) and [Watershed](https://en.wikipedia.org/wiki/Watershed_(image_processing)) to extract the edges of the cells and segment them from the microscope images, and also get the centroid and area of each cell. 

## Procedure

Take polyurethane microscope image as an example. At first, we used Watershed directly to segment each cell and get its centroid. But the result is not as expected because some cells has the irregular shape and unclear edge. We had to adjust the parameter manually image by image, which is not repeatable work. So we tried to investigate some deep learning algorithm to automatically extract the edge.

We decided to use Unet to extract the edge of the cell because of a similar scenario: [ISBI Neural Electron Microscope Image Segmentation challenge](http://brainiac2.mit.edu/isbi_challenge/home). A [github sample](https://github.com/zhixuhao/unet) tried to use UNET to extract the edge of neural electron and got very good accuracy after little training steps. The ISBI challenge is as below:
<p align="center"><img src="unet+watershed_research_code/image/isbi.jpg" width="40%" height="40%">

So the final step is 
1) transform the original image to contrast gray image
2) Manually label several typical images as black and white edge image
3) train the whole UNET model, or with transfer learning 
4) transform the edge extraction result to binary image
5) Use Watershed to extract all the cells from the edge result and get centroid and area information

For labeling the edge image, we tried two approaches:
1) Use [Labelme](https://github.com/wkentaro/labelme) to label the cell by polygon and generate the black and white edge image based on the json file, with white color inside cell and black color filled the edge.
2) Use default Painting app to draw the edge manually.


## code description
Use UNET implementation from the [github sample from zhixuhao]((https://github.com/zhixuhao/unet)), upload the image folder to */data* folder with a train folder and a test folder, each folder will cover a image folder and a label folder, each includes the .png file named from 0.

<p align="center"><img src="unet+watershed_research_code/image/folder.jpg" width="30%" height="30%">

The data is used for transfer learning based on a pre-trained Unet model ( like *unet_membrane.hdf5* ). You can leverage the Azure Machine Learning script in the folder of [azureml_script](azureml_script/aml_devops.md)  to train the model and download it.

Last step, use Skimage implemented Watershed algorithm to extract each cell and get its centroid and area.


## Train and deploy Unet on Keras model using Azure Machine Learning Service

### Introduction

In the folder of [unet+watershed_research_code](unet+watershed_research_code/cell_extraction.md), we described the solution of extracting cell from chemical or biological microscope images. This folder will show how to leverage Azure Machine Learning Service Python SDK to train, and deploy Unet on Keras model in Azure Container Instance.

### Azure Machine Learning Service

[Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/) is a cloud service that you use to train, deploy, automate, and manage machine learning models, all at the broad scale that the cloud provides. Besides portal UI, it also provides Python SDK to improve productivity.

<p align="center"><img src="azureml_script/image/aml_architecture.png" width="80%" height="80%">

### Code description

The data and Unet python script we used for training is from [github](https://github.com/zhixuhao/unet), actually sample dataset is from a [ISBI challenge](http://brainiac2.mit.edu/isbi_challenge/). The script including *trainUnet.py*, *data.py*, *model.py* are in aml_script folder.

In score script, to enable batch prediction, we accept the input of Azure Storage Container url where stores all the test images, and save the prediction images to another container (you should config its location in the *score.py* file), then return the result container url. So you can download them from the container to local computer. If single input is needed, code in score file must be changed.

