import os
import glob
import torch
import torchvision
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