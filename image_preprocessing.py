import os
import glob
import shutil
from PIL import Image
import torch, torchvision

data_path = "data_final/"

# Splitting the dataset

augmented_datasets = torchvision.datasets.ImageFolder(root = augmented_data_path, transform = transform)
train_dataset, valid_dataset = torch.utils.data.random_split(augmented_datasets, [train_size, valid_size])

def move_files(file_list, source_directory, destination_directory):
    for filename in file_list:
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, filename)
        shutil.move(source_path, destination_path)




highres_path = "data_final/highres/"
lowres_path = "data_final/lowres/"

def downscale(image_path, output_path, scale):
    with Image.open(image_path) as img:
        small = img.resize((img.size[0] // scale, img.size[1] // scale))
        result_img = small.resize(img.size)
        result_img.save(output_path)



pattern = os.path.join(image_path, '*')
image_files = glob.glob(pattern)
for image_file in image_files:
    downscale(image_file, os.path.join(output_path, os.path.basename(image_file)), scale=5)