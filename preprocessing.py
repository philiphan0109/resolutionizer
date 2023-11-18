import os
import glob
import torch
import torchvision
from torchvision import datasets, transforms
import tkinter as tk
from PIL import Image, ImageGrab, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


# Creating lowres images
image_path = "data_final/highres/"
output_path = "data_final/lowres/"

def downscale(image_path, output_path, scale):
    with Image.open(image_path) as img:
        small = img.resize((img.size[0] // scale, img.size[1] // scale))
        result_img = small.resize(img.size)
        result_img.save(output_path)



pattern = os.path.join(image_path, '*')
image_files = glob.glob(pattern)
for image_file in image_files:
    downscale(image_file, os.path.join(output_path, os.path.basename(image_file)), scale=5)


# Splitting the dataset (maybe do this in train?)
data_path = "data_final/"
dataset = torchvision.datasets.ImageFolder(root = data_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size= 32, shuffle = True)

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

# Display class names and index labels for debug
print("Class names: ", dataset.classes)
print("Class index mapping:", dataset.class_to_idx)

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
print(f"Total Images: {len(dataset)}")
print(f"Training Images: {len(train_dataset)}")
print(f"Validation Images: {len(valid_dataset)}")

