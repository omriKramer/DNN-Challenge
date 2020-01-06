import torch
from torch import nn


def batch_meals(meals):
    max_size = max(len(indv_meals) for indv_meals in meals)
    batched_meals_shape = len(meals), max_size, meals[0].shape[1]
    batched_meals = meals[0].new_zeros(batched_meals_shape)
    for indv_meals, pad_meals in zip(meals, batched_meals):
        pad_meals[:indv_meals.shape[0]].copy_(indv_meals)

    return batched_meals


class BranchModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.meals_branch = MealsModel()
        self.cgm_branch = CGMModel()
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )

    def forward(self, cgm, meals):
        meals = batch_meals(meals)
        meals_features = self.meals_branch(meals)
        cgm_features = self.cgm_branch(cgm)
        features = torch.cat((meals_features, cgm_features), dim=1)
        pred = self.head(features)
        return pred


class MealsModel(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        last_dim = 40
        for d in [128, 256, 512, 512]:
            layers.extend([nn.Linear(last_dim, d), nn.ReLU()])
            last_dim = d
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        out = out.mean(dim=1)
        return out


class CGMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 3),)
        self.bn1 = nn.BatchNorm1d(64, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(128, 128)
        self.conv3 = nn.Conv1d(128, 216, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(216, 216)
        self.conv4 = nn.Conv1d(216, 512, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm1d(512, 512)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x[:, None, ...]
        out = self.conv1(x)
        out = out.squeeze(dim=2)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.avg_pool(out)
        out = out.squeeze()
        return out