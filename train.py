import os
import torch
import torch.nn as nn
from model import SRCNN
import torchvision.transforms as transforms
from PIL import Image

model = SRCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

lr_file_names = [f for f in os.listdir("data_final/lowres") if f.endswith(".JPEG")]
hr_file_names = [f for f in os.listdir("data_final/highres") if f.endswith(".JPEG")]

transform = transforms.Compose([transforms.ToTensor()])

num_epochs = 1
for epoch in range(num_epochs):
    for lr_file, hr_file in zip(lr_file_names, hr_file_names):
        lr = transform(Image.open(os.path.join("data_final/lowres", lr_file)).convert('RGB')).unsqueeze(0)
        hr = transform(Image.open(os.path.join("data_final/highres", hr_file)).convert('RGB')).unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(lr)
        loss = criterion(outputs, hr)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

