"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils,loss_and_accuracy_curve_plotter,testing


from torchvision import transforms
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 8
HIDDEN_UNITS = 20
LEARNING_RATE = 0.001


#going modular/data/pizza_steak_sushi/train
# Setup directories
root_dir = "/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/data/data/tennessee/2018/percentiles/H0H1-3 channels"
annotation_file_path = "/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/data/data/tennessee/2018/SVI2018 TN counties with death rate HepVu/SVI2018_TN_counties_with_death_rate_HepVu.shp"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, validation_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    annotation_file_path=annotation_file_path,
    root_dir=root_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

start_time = timer()

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             validation_dataloader=validation_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             use_mixed_precision=True,
             save_name="tinyvgg_model.pth",
             save_path="/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/models/")

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

#plotting the results
loss_and_accuracy_curve_plotter.plot_loss_curves(results)

# Test the model after training
test_loss, test_acc = testing.test_step(model=model,
                                  dataloader=test_dataloader,
                                  loss_fn=loss_fn,
                                  device=device,
                                  use_mixed_precision=True)

# Print out test results
print(
    f"Test results | "
    f"test_loss: {test_loss:.4f} | "
    f"test_acc: {test_acc:.4f}"
)

# Update results dictionary with test results
results["test_loss"].append(test_loss)
results["test_acc"].append(test_acc)
