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
#Get all datasets from the excel file
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
useful = ['id','observed_on','image_url','num_identification_agreements','latitude','longitude','field:phenology (foa)']

flowers_one= flowers_base[useful]
buds_one = buds_base[useful]
fruit_one = fruit_base[useful]
No_Evidence_one = No_Evidence_Base[useful]

flowers_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
buds_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
fruit_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)
No_Evidence_one.rename(columns={'field:phenology (foa)':'phenology'}, inplace=True)

# %%
flowers_one["phenology"]=flowers_one["phenology"].fillna("Flowers")
buds_one["phenology"]=buds_one["phenology"].fillna("Buds")
fruit_one["phenology"]=fruit_one["phenology"].fillna("Fruit")
No_Evidence_one["phenology"]=No_Evidence_one["phenology"].fillna("No Evidence")

# %%
from sklearn.model_selection import train_test_split

train_flowers, test_flowers = train_test_split(flowers_one, test_size=0.25, random_state=42)
train_buds, test_buds = train_test_split(buds_one, test_size=0.25, random_state=42)
train_fruit, test_fruit = train_test_split(fruit_one, test_size=0.25, random_state=42)
train_No_Evidence, test_No_Evidence = train_test_split(No_Evidence_one, test_size=0.25, random_state=42)

# %%
test_flowers.head()
#test_buds.head()
#test_fruit.head()
#test_No_Evidence.head()

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
Train_set = Train_set.drop_duplicates()
Test_set = Test_set.drop_duplicates()

# %%
Train_set[Train_set.duplicated()]

# %%
print(Train_set.describe([x*0.1 for x in range(10)]))
print(Test_set.describe([x*0.1 for x in range(10)]))

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
    executor.map(lambda x: download_image(x[1], x[0], "Train"), enumerate(lastTrain_set['image_url'], start=0))
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda x: download_image(x[1], x[0], "Test"), enumerate(Test_set['image_url'], start=0))

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

def extract_number(file):
    # Using regex to find numbers in the image file
    match = re.search(r'(\d+)', file)
    return int(match.group(0)) if match else float('inf')

# %%
Test_image_list = []
Train_image_list = []
SIZE = 224

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


# %%
label_map = {
    'Flowers': 0,
    'Buds': 1,
    'Fruit': 2,
    'No Evidence': 3,
    'Buds & Flowers': 4,
    'Flowers & Buds': 5,
    'Flowers & Fruit': 6,
    'Flower, Buds & Fruit': 7
}

# %%
processedDF["phenology"] = processedDF["phenology"].map(label_map).astype(int)
processedDF.head()

# %%

# Image transformations (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
num_classes = len(processedDF['phenology'].unique())  # Number of unique classes in the dataframe
model = SingleLabelCNN(num_classes).to(device)

# %%
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy for single-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training loop
epochs = 15
for epoch in range(epochs):
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
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# %%
model.eval()

# Initialize counters for accuracy
correct = 0
total = 0

# Turn off gradients
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
print(f'Accuracy of the model on the train dataset: {accuracy:.2f}%')

# %% [markdown]
# __Save model to torch file__

# %%
torch.save(model, 'final_model2.pth')

# %% [markdown]
# __Load model from torch file__

# %%
loaded_model = torch.load('final_model2.pth')
loaded_model.eval()

# %%
def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


# %%
# Path to the image you want to classify
image_path = 'Test/367.png'

# Preprocess the image
img_tensor = preprocess_image(image_path, target_size=(224, 224))

# Predict the class
with torch.no_grad():  # Disable gradient tracking
    output = loaded_model(img_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

print(f'Predicted class: {predicted_class}')


# %%
testDF = pd.DataFrame({'image': Test_image_list,
    'phenology': Test_Labels})
testDF['phenology'] = testDF['phenology'].map(label_map).astype(int)

# %%
correct_outputs = 0

for _, row in testDF.iterrows():
    # Preprocess each image
    img_tensor = preprocess_image(row['image'])
    
    # Make prediction
    with torch.no_grad():  # No need for gradients during inference
        output = loaded_model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Compare with the true label
    if predicted_class == row['phenology']:
        correct_outputs += 1

# Calculate accuracy
accuracy = correct_outputs / len(testDF)
print(f'Accuracy: {accuracy * 100:.2f}%')



