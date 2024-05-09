"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils,loss_and_accuracy_curve_plotter


from torchvision import transforms
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


#going modular/data/pizza_steak_sushi/train
# Setup directories
train_dir = "/Users/h6x/ORNL/git/persistence-image-classification/going modular/data/pizza_steak_sushi/train"
test_dir = "/Users/h6x/ORNL/git/persistence-image-classification/going modular/data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
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
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

#plotting the results
loss_and_accuracy_curve_plotter.plot_loss_curves(results)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="/Users/h6x/ORNL/git/persistence-image-classification/models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")