"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils,loss_and_accuracy_curve_plotter,testing
import torchvision


from torchinfo import summary
from torchvision import transforms
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 40
LEARNING_RATE = 0.0001


#going modular/data/pizza_steak_sushi/train
# Setup directories
root_dir = "/Users/h6x/ORNL/git/persistence-image-classification/data/tennessee/2018/percentiles/below 90/h1/npy 3 channels"
annotation_file_path = "/Users/h6x/ORNL/git/persistence-image-classification/data/tennessee/2018/SVI2018 TN counties with death rate HepVu/SVI2018_TN_counties_with_death_rate_HepVu.shp"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
# weights = torchvision.models.MobileNet_V2_Weights.DEFAULT # .DEFAULT = best available weights 
# model = torchvision.models.mobilenet_v2(weights=weights).to(device)


# # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
# for param in model.features.parameters():
#     param.requires_grad = False

# Whatever the tranfer learning model that we select, using auto transforms we can get the transforms(less customization)
auto_transforms = weights.transforms()

# # Create transforms
# data_transform = transforms.Compose([
#   transforms.Resize((64, 64)),
#   transforms.ToTensor()
# ])

# Create DataLoaders with help from data_setup.py
train_dataloader, validation_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    annotation_file_path=annotation_file_path,
    root_dir=root_dir,
    transform=auto_transforms,
    batch_size=BATCH_SIZE
)


# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)


# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)

# # Print a summary using torchinfo (uncomment for actual output)
# summary(model=model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ) 


# # Create model with help from model_builder.py
# model = model_builder.TinyVGG(
#     input_shape=3,
#     hidden_units=HIDDEN_UNITS,
#     output_shape=len(class_names)
# ).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# optimizer = torch.optim.SGD(model.parameters(),lr=1e-3, momentum=0.9)


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
             save_path="/Users/h6x/ORNL/git/persistence-image-classification/transfer learning - model 1/models/")

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