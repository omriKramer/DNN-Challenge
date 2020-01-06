import utils
import transform
from datasets import GlucoseData
from torch.utils.data import DataLoader
import torch
from torch import nn
import models

cgm_file, meals_file = utils.get_files()
train, val = GlucoseData.train_val_split(cgm_file, meals_file, transform=transform.to_tensor)
train_dl = DataLoader(train, batch_size=128, shuffle=True, collate_fn=utils.collate, num_workers=8)
val_dl = DataLoader(val, batch_size=128, collate_fn=utils.collate, num_workers=8)
model = models.BranchModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)


def main():
    for epoch in range(2):
        running_loss = 0.0
        model.train()
        for i, samples in enumerate(train_dl):
            cgm = samples['cgm']
            meals = samples['meals']
            target = samples['target']

            optimizer.zero_grad()
            out = model(cgm, meals)
            loss = criterion(out, target)
            if not torch.isfinite(loss):
                print(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        val_loss = evaluate()
        print(f'[{epoch + 1}] val loss: {val_loss:.3f}')


@torch.no_grad()
def evaluate():
    running_loss = 0.0
    for i, samples in enumerate(val_dl):
        cgm = samples['cgm']
        meals = samples['meals']
        target = samples['target']

        out = model(cgm, meals)
        loss = criterion(out, target)
        running_loss += loss * len(meals)

    return running_loss / len(val_dl.dataset)


main()
