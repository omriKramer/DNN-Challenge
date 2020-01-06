from torch import nn


class Mirror(nn.Module):

    def forward(self, cgm, meals):
        changes = cgm[:, -9:-1, 1] - cgm[:, -1, 1]
        out = changes.flip(1)
        return out
