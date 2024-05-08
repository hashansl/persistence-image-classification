# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from PIL import Image


from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches


from CustomDataset_peristence_img import data_loader_persistence_img
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 6
learning_rate = 3e-5
batch_size = 16
num_epochs = 100


# Load data
custom_dataset = data_loader_persistence_img(geo_file_path='/Users/h6x/ORNL/git/opioid-risk-modeling/tennessee/data/processed data/SVI2018 TN counties with death rate HepVu/SVI2018_TN_counties_with_death_rate_HepVu.shp',
    root_dir='/Users/h6x/ORNL/git/opioid-risk-modeling/tennessee/results/persistence images/percentiles/H0H1/EP_POV', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(custom_dataset, [70, 25])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Model
# model alexnet
# model = torchvision.models.mobilenet_v2(weights="DEFAULT")
model = torchvision.models.googlenet(weights="DEFAULT")

# freeze all layers, change final linear layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer is not frozen
model.fc = nn.Linear(in_features=1024, out_features=num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

import matplotlib.pyplot as plt

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Train Network
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for the epoch
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    # Print training and validation loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plotting
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
