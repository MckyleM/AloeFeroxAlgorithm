# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import DataFrame, Series
import seaborn as sns
import urllib.request
from PIL import Image
import concurrent.futures
from collections import OrderedDict
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import tensorflow as tf
import cv2
import requests
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

import glob
import os
#from skimage import img_as_ubyte

# %%
Base_Aloe_Ferox_Dataset = pd.ExcelFile('BaseSet.xlsx')

# %%

Base_Set = {}
for sheet in Base_Aloe_Ferox_Dataset.sheet_names:
    df = Base_Aloe_Ferox_Dataset.parse(sheet)
    Base_Set[sheet] = df

flowers_base = Base_Set["FLOWERS"]
buds_base = Base_Set["BUDS"]
fruit_base = Base_Set["FRUIT"]
No_Evidence_Base = Base_Set["No Evidence"]

flowers_base.head()

# %%
buds_base.head()

# %%
fruit_base.head()

# %%
No_Evidence_Base.head()

# %%
def missing_values(df):
    print(f'Column\t\t\t% missing')
    print(f'{"-"}'*35) 
    return (df.isnull().sum()/len(df)*100
    ).astype(int)

missing_values(flowers_base)

# %%
#all null columns or columns with 100% the same value have been removed
useful = ['id','time_observed_at','image_url','num_identification_agreements','latitude','longitude','field:phenology (foa)']

flowers_one= flowers_base[useful]
buds_one = buds_base[useful]
fruit_one = fruit_base[useful]
No_Evidence_one = No_Evidence_Base[useful]

flowers_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
buds_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
fruit_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
No_Evidence_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)

# %%
train_flowers = flowers_one[flowers_one.phenology.isnull()]
train_buds = buds_one[buds_one.phenology.isnull()]
train_fruit = fruit_one[fruit_one.phenology.isnull()]
train_No_Evidence = No_Evidence_one[No_Evidence_one.phenology.isnull()]

# %%
test_flowers = flowers_one[flowers_one.phenology.notnull()]
test_buds = buds_one[buds_one.phenology.notnull()]
test_fruit = fruit_one[fruit_one.phenology.notnull()]
test_No_Evidence = No_Evidence_one[No_Evidence_one.phenology.notnull()]

# %%
test_flowers.head()
#test_buds.head()
#test_fruit.head()
#test_No_Evidence.head()

# %%
train_flowers.loc[:, "phenology"]="Flowers"
train_buds.loc[:, "phenology"]="Buds"
train_fruit.loc[:, "phenology"]="Fruit"
train_No_Evidence.loc[:, "phenology"]="No Evidence"

# %%
Test_set = pd.concat([test_flowers, test_buds, test_fruit, test_No_Evidence], axis=0)
Train_set = pd.concat([train_flowers, train_buds, train_fruit, train_No_Evidence], axis=0)

Train_set.info()

# %%
missing_values(Train_set)

# %%
Train_set = Train_set.dropna()

Train_set.info()

# %%
missing_values(Test_set)

# %%
#Train_set.sort_values(by='phenology')

# %%
#Test_set.sort_values(by='phenology')

# %%
Train_set[Train_set.duplicated()]

# %%
Train_set.describe([x*0.1 for x in range(10)])
Test_set.describe([x*0.1 for x in range(10)])

# %%
sns.displot(data =Train_set['latitude'])
sns.displot(data =Train_set['longitude'])

# %%
print(Test_set.index)

# %%
Train_set = Train_set.reset_index(drop=True)
Test_set = Test_set.reset_index(drop=True)

# %%
Test_set.info()

# %%
missing_values(Train_set)

# %%
Train_set.info()

# %%
lastTrain_set = Train_set[['image_url','phenology']].copy()
lastTrain_set.head()

# %%
def download_image(image_url, image_name_counter, folder):
    file_name = f"./{folder}/{image_name_counter}.png"
    urllib.request.urlretrieve(image_url, file_name)
    print(f"Downloaded {file_name}")

# %%
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda x: download_image(x[1], x[0], "Train"), enumerate(lastTrain_set['image_url'], start=1))
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda x: download_image(x[1], x[0], "Test"), enumerate(Test_set['image_url'], start=1))

# %%
Train_set.head()

# %%
Train_labels = []
Test_Labels = []
for i in range(len(lastTrain_set)-5):
    Train_labels.append(lastTrain_set['phenology'][i])

for i in range(len(Test_set)):
    Test_Labels.append(Test_set['phenology'][i])

print(Train_labels)
print(len(Train_labels))

# %%
import re

def extract_number(filename):
    # Use regex to find numbers in the filename
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')

# %%
Test_image_list = []
Train_image_list = []
SIZE = 224
#Change path back to main folders
path_test = r".\Test\*.png"
path_train = r".\Train\*.png"

Test_image_list = sorted(glob.glob(path_test),key=lambda x: extract_number(os.path.basename(x)))
Train_image_list = sorted(glob.glob(path_train),key=lambda x: extract_number(os.path.basename(x)))

#Test_image_list = np.array(Test_image_list)
#Train_image_list  = np.array(Train_image_list)

# %%
Test_image_list
print(len(Train_image_list))

# %%
processedDF = pd.DataFrame({'image': Train_image_list,
                            'phenology': Train_labels})

# %%
processedDF.head()

# %%
processedDF['phenology'] = processedDF['phenology'].astype('category').cat.codes

# %%

# Image transformations (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        path = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['phenology']
        
        # load image
        image = Image.open(path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

# Initialize dataset and dataloader
dataset = ImageDataset(dataframe=processedDF, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class SingleLabelCNN(nn.Module):
    def __init__(self, num_classes):
        super(SingleLabelCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # Adjust for single label output

    def forward(self, x):
        x = self.base_model(x)
        return nn.functional.log_softmax(x, dim=1)  # Use softmax for single-label classification

# Instantiate the model with the number of classes
num_classes = len(processedDF['phenology'].unique())  # Number of unique classes in your dataset
model = SingleLabelCNN(num_classes).to(device)

# %%
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy for single-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for image, label in dataloader:
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# %%
model.eval()

# Initialize counters for accuracy
correct = 0
total = 0

# Turn off gradients to save memory and computation
with torch.no_grad():
    for images, labels in dataloader:
        # Move data to the appropriate device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: get predictions
        outputs = model(images)
        
        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs, 1)
        
        # Update total and correct counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

# %% [markdown]
# __Save model to pickle file__

# %%
import pickle

model_file = 'final_model.pkl'
pickle.dump(model, open(model_file,'wb'))

# %% [markdown]
# __Load model from pickle file__

# %%
loaded_model = pickle.load(open(model_file,'rb'))


