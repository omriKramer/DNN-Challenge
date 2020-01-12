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

    def forward(self, cgm, meals_cont, meals_cat):
        meals_cont = batch_meals(meals_cont)
        meals_cat = batch_meals(meals_cat)
        meals_features = self.meals_branch(meals_cont, meals_cat)
        cgm_features = self.cgm_branch(cgm)
        features = torch.cat((meals_features, cgm_features), dim=1)
        pred = self.head(features)
        return pred


class MealsModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.food_emb = nn.Embedding(5830, 206)
        self.unit_emb = nn.Embedding(90, 20)
        self.meal_emb = nn.Embedding(5, 4)
        layers = []
        last_dim = 37 + 230
        for d in [128, 256, 256]:
            layers.extend([nn.Linear(last_dim, d), nn.ReLU()])
            last_dim = d
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 256))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 256))

    def forward(self, cont, cats):
        x = cont, self.food_emb(cats[..., 0]), self.meal_emb(cats[..., 1]), self.unit_emb(cats[..., 2])
        x = torch.cat(x, dim=-1)
        out = self.layers(x)
        out = torch.cat((self.avg_pool(out), self.max_pool(out)), dim=2)
        out.squeeze_(dim=1)
        return out


class CGMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 3), )
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
