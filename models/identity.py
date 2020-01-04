from torch import nn


class Identity(nn.Module):

    def forward(self, cgm, meals):
        last_value = cgm[:, 1, -1]
        out = last_value.repat(last_value.shape[0], 8)
        return out
