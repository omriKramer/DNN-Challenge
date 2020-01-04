from scipy.stats import pearsonr
import numpy as np
import transform
import utils
from models import Mirror
from datasets import GlucoseData
from torch.utils.data import DataLoader


cgm_file, meals_file = utils.get_files()
train, val = GlucoseData.train_val_split(cgm_file, meals_file, transform=transform.to_tensor)
val_dl = DataLoader(val)
model = Mirror()

amount = 0
acc_pearson = 0.0
for sample in val_dl:
    out = model(sample['cgm'], sample['meals'])
    out = out.numpy()[0]
    target = sample['target'].numpy()[0]
    corr, _ = pearsonr(out, target)
    if not np.isnan(corr):
        amount += 1
        acc_pearson += corr

total = acc_pearson / amount
print(f'{total:.2f} mean pearson correlation')
print(f'{len(val) - amount}/{len(val)} were nans')
