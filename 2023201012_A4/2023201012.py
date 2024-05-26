import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.models as models



# ...........................................................................................Data Preparation................................................................................ 

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)


train_path = '/kaggle/input/age-predict/content/faces_dataset/train'
train_ann = '/kaggle/input/age-predict/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = '/kaggle/input/age-predict/content/faces_dataset/test'
test_ann = '/kaggle/input/age-predict/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

validation_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - validation_size

# Split the training dataset into training and validation sets
train_subset, val_subset = random_split(train_dataset, [train_size, validation_size])

# Create DataLoader instances for training, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# ................................................................resnet34 Model training part .....................................................................................................

best_val_loss = float('inf')

# Load pre-trained ResNet34 model
resnet_model = models.resnet34(weights='IMAGENET1K_V1')

# Modify the last fully connected layer for age prediction
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 1)  # Outputs 1 value for age prediction

# Define optimizer and loss function
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001,momentum=0.9)
criterion = nn.L1Loss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)

num_epochs = 30
for epoch in range(num_epochs):
    resnet_model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet_model(inputs)
        
        labels = labels.view(-1, 1) if labels.dim() == 1 else labels
        
        loss = criterion(outputs, labels)
        
#         # Calculate absolute difference between predictions and labels
#         abs_diff = torch.abs(outputs - labels)
        
#         # Create mask: 0 where abs_diff <= 2, 1 otherwise
#         mask = (abs_diff > 2).float()
        
#         # Apply mask to loss calculation
#         loss = torch.mean(criterion(outputs, labels) * mask)

        
        # Compute gradients and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}')

    # Validation
    resnet_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = resnet_model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

#     # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(resnet_model.state_dict(), 'best_model.pt')

print('Training finished.')

# ............................................. getting predictions using this resnet34 model .........................................................................................

resnet_model = models.resnet34(weights='IMAGENET1K_V1')

# Modify the last fully connected layer for age prediction
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 1)  # Output 1 value for age prediction

# Load the saved model's state dictionary
saved_model_path = '/kaggle/working/best_model.pt'
resnet_model.load_state_dict(torch.load(saved_model_path))
resnet_model.to(device)

# Set the model to evaluation mode

@torch.no_grad
def predict(loader):
    resnet_model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = resnet_model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

preds = predict(test_loader)

submit = pd.read_csv('/kaggle/input/age-predict/content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('baseline.csv',index=False)



# ....................Extracting features from the above trained model and using it to predict the age with the help of SVR.......................................

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Modify the last fully connected layer for age prediction
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 1)  # Output 1 value for age prediction

# Load the saved model's state dictionary
saved_model_path = '/kaggle/working/best_model.pt'
resnet_model.load_state_dict(torch.load(saved_model_path))
resnet_model.to(device)
resnet_model.fc = nn.Identity()  # Remove last fully connected layer

def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            output = model(inputs).detach().cpu().numpy()
            features.extend(output)
            labels.extend(target.numpy())
    return features, labels

train_features, train_labels = extract_features(resnet_model, train_loader)
# Train SVR model
svr_model = SVR(kernel='rbf')
svr_model.fit(train_features, train_labels)
@torch.no_grad
def predict(loader, model):
    resnet_model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

#         pred = resnet_model(img)
        outputs_resnet = resnet_model(img).squeeze().cpu().numpy()
    
        outputs_svr = svr_model.predict(outputs_resnet)
        predictions.extend(outputs_svr)

    return predictions

preds = predict(test_loader, svr_model)

submit = pd.read_csv('/kaggle/input/age-predict/content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('baseline_svr.csv',index=False)

# ................................................Now using the both models predictions to get the final results...............................................

import pandas as pd
df1 = pd.read_csv('/content/baseline.csv')
df2 = pd.read_csv('/content/baseline_svr.csv')
column_name = 'age'  
df1_column = df1[column_name]
df2_column = df2[column_name]
average_column = (df1_column + df2_column) / 2
df1[column_name] = average_column
df1.to_csv('final_result.csv', index=False)

