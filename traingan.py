import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
from model import Generator, Discriminator
from PIL import Image

batch_size = 1

real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)
num_epochs = 1
criterion = nn.BCELoss()

lr_file_names = [f for f in os.listdir("data_final/lowres") if f.endswith(".JPEG")]
hr_file_names = [f for f in os.listdir("data_final/highres") if f.endswith(".JPEG")]

transform = transforms.Compose([transforms.ToTensor()])

for epoch in range(num_epochs):
    for lr_file, hr_file in zip(lr_file_names, hr_file_names):
        lr = transform(Image.open(os.path.join("data_final/lowres", lr_file)).convert('RGB')).unsqueeze(0)
        hr = transform(Image.open(os.path.join("data_final/highres", hr_file)).convert('RGB')).unsqueeze(0)


        outputs_real = Discriminator(real_images)
        print(outputs_real)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
