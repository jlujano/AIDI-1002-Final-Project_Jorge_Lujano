# AIDI-1002-Final-Project_Jorge_Lujano

## AIDI 1002 Final Project: Object Detection with Faster R-CNN

This repository contains the code where we compare the performance of a custom-trained Faster R-CNN model against a YOLOv8 model for object detection on the VisDrone dataset.

## Project Description
The goal of this project is to implement and fine-tune a Faster R-CNN model on the VisDrone dataset, a challenging aerial imagery dataset. The model's performance will be evaluated on key metrics like mean Average Precision (mAP) and compared against a pre-trained or fine-tuned YOLOv8 model to analyze the trade-offs between speed and accuracy for drone-based object detection.

## Code Structure
AIDI_1002_Final_Project.ipynb: The main Jupyter Notebook containing all the code for data loading, model setup, training, and evaluation.

## How to Run the Code
### Prerequisites

* Python: Ensure you have Python 3.8 or newer installed.
* VisDrone Dataset: Download the VisDrone2019-DET dataset from the official website and place the VisDrone2019-DET-train and VisDrone2019-DET-val folders inside a parent directory.


### Environment Setup

* Create a virtual environment (recommended):
python -m venv yolo_env
yolo_env\Scripts\activate

* Install the required libraries:

  *from pathlib import Path
  
  *from PIL import Image
  
  *from tqdm import tqdm
  
  *from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
  
  *from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

* Update the root_dir variable in the notebook to point to the location of your VisDrone dataset. 

* Run all the cells in the notebook to train the Faster R-CNN model, evaluate its performance, and save the trained model weights.

* Running on a GPU (Highly Recommended):
The code is already configured to use a GPU if available
