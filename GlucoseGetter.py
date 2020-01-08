#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, random_split
from datasets.glucose_meals_v2 import GlucoseData, split_by_individuals
from train import train_model
from radam import RAdam
from optimizer import Lookahead
from ranger import Ranger
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from transform.transform import combine_cgm_meals
from models.seq2seq import Seq2Seq
from radam import RAdam
from optimizer import Lookahead

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Pad((32, 16)),
#     torchvision.transforms.ToTensor(),
# ])

LOAD_STATE = True
TEST_STATE = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Glucose_data
glucose_file = 'data/GlucoseValues.csv'
meal_file = 'data/Meals.csv'

lossFunc = nn.MSELoss()


train, val = split_by_individuals(glucose_file, meal_file, ratio=0.8)
glucose_train = GlucoseData(train, transform=combine_cgm_meals)
glucose_val = GlucoseData(val, transform=combine_cgm_meals)


loaders_dict = {
        'train': torch.utils.data.DataLoader(glucose_train, batch_size=400, shuffle=False, drop_last=True),
        'val': torch.utils.data.DataLoader(glucose_val, batch_size=400, shuffle=False, drop_last=True)
    }


model = Seq2Seq()
model.to(device)

optimizer_base = RAdam(model.parameters(), lr=1e-1)
optimizer = Lookahead(optimizer=optimizer_base, k=5, alpha=0.5)


model.train()

train_model(model=model, dataloaders=loaders_dict, optimizer=optimizer, device=device, num_epochs=25)


