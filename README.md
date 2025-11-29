# CNN Implementation for Plastic Object Detection

A convolutional-neural-network based classification & detection project to identify various types of plastic waste (cutlery, lids, bottles, bags, straws etc.).  
Built with Python, Keras/TensorFlow and YOLO integration.

## 🧩 Project Overview  
This repository contains:  
- A CNN model (`plastic_cnn.h5` + `plastic_cnn.json`) that classifies images of plastic waste into categories.  
- A YOLO dataset (`yolo_dataset/`) and model (`yolov8n.pt`) for object detection of plastic items in images.  
- A Flask/Streamlit (or whichever) app (`app.py`) that allows you to upload images and get classification/detection results.  
- Training notebook (`train.ipynb`) to show dataset processing, model training, evaluation.  
- Label encoder (`label_encoder.pkl`) to map class indices to category names.  
- Organized folder structure for datasets and results (`plastic_images/`, `saved_results/`).

## ✅ Features  
- Multi-class classification of plastic items: e.g., disposable cutlery, plastic cup lids, detergent bottles, shopping bags, polyethene bags, soda bottles, straws, trash bags, water bottles.  
- Detection pipeline with YOLO for locating plastic items in images.  
- Easy-to-use interface to test new images and view results.  
- Modular structure enabling re-training, fine-tuning or deployment.

## 📂 Repository Structure  

##code image
<img width="2879" height="1668" alt="Screenshot 2025-11-29 110650" src="https://github.com/user-attachments/assets/85556401-0d7a-4243-96f1-8cd676b5eff3" />

