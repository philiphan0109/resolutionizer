import os
import glob
import shutil

data_path = "imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"
output_path = "data/"

for sub_folder in next(os.walk(data_path))[1]:
    path = os.path.join(data_path, sub_folder)
    files = glob.glob(os.path.join(path, ""))[:10]
    
    for file in files:
        shutil.move(file, "~/data")