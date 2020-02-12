import torch
from torch import nn


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn=None):
    """Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."""
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class Linear(nn.Module):

    def __init__(self, emb_drop=0.5):
        super().__init__()
        self.n_cont = 12 * 36 + 50
        self.food_emb = nn.Linear(5830, 200)
        self.unit_emb = nn.Linear(88, 20)
        self.meal_emb = nn.Linear(5, 4)
        self.n_emb = 12 * (200 + 20 + 4)
        self.bn_cont = nn.BatchNorm1d(self.n_cont)
        self.emb_drop = nn.Dropout(emb_drop)
        sizes = [self.n_emb + self.n_cont, 1024, 512, 256, 128, 64, 8]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [None]
        ps = [0.] + [0.5] * (len(sizes) - 2)
        bns = [False, ] + [True] * (len(sizes) - 3) + [False]
        layers = []
        for n_in, n_out, p, b, act in zip(sizes[:-1], sizes[1:], ps, bns, actns):
            layers += bn_drop_lin(n_in, n_out, bn=b, p=p, actn=act)
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        x_cat = self.food_emb(sample['food_id']), self.unit_emb(sample['unit_id']), self.meal_emb(sample['meal_type'])
        x_cat = torch.cat(x_cat, dim=-1).flatten(start_dim=1)
        x_cat = self.emb_drop(x_cat)
        x_cont = torch.cat((sample['cgm'], sample['cont'].flatten(start_dim=1)), dim=-1)
        x = torch.cat((x_cont, x_cat), dim=-1)
        x = self.layers(x)
        return x
