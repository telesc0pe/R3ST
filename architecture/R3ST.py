import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from architecture.MST_Plus_Plus import MST_Plus_Plus


class MST(nn.Module):

    def __init__(self):
        super(MST, self).__init__()
        self.Spectral_wise_Transformer = MST_Plus_Plus(stage=1)

    def forward(self, x):
        temp = x
        x = (x - x.min()) / (x.max() - x.min())
        return temp, self.Spectral_wise_Transformer(x)[1]


class U3SM_ST(nn.Module):

    def __init__(self, N = 2):
        super(U3SM_ST, self).__init__()
        self.U3SM = MST_Plus_Plus(stage=1)
        self.Spectral_wise_Transformer = MST_Plus_Plus(in_channels=3 * N * 3, stage=1)
        self.N = N

    def forward(self, x):
        soft_seg = self.U3SM(x)[1]
        soft_seg = soft_seg[:, :3 * self.N, :, :]
        input = x.repeat(1, self.N, 1, 1)
        masks = torch.cat((input, soft_seg * input, (1 - soft_seg) * input), dim=1)
        hypers = self.Spectral_wise_Transformer(masks)[1]
        return soft_seg[:, :3, :, :], hypers


class RREM_ST(nn.Module):

    def __init__(self):
        super(RREM_ST, self).__init__()
        self.RREM = MST_Plus_Plus(stage=1)
        self.Spectral_wise_Transformer = MST_Plus_Plus(in_channels=6, stage=1)

    def forward(self, input):
        x = self.RREM(input)[1][:, :3, :, :] + input
        return x, self.Spectral_wise_Transformer(torch.cat((x, input), dim=1))[1]


class R3ST(nn.Module):

    def __init__(self, N = 2):
        super(R3ST, self).__init__()
        self.RREM = MST_Plus_Plus(stage=1)
        self.U3SM = MST_Plus_Plus(in_channels=6, stage=1)
        self.Spectral_wise_Transformer = MST_Plus_Plus(in_channels=3 * N * 3, stage=1)
        self.N = N

    def forward(self, rgb):
        r = self.RREM(rgb)[1][:, :3, :, :] + rgb
        soft_seg = self.U3SM(torch.cat((r, rgb), dim=1))[1]
        input = r.repeat(1, self.N, 1, 1)
        soft_seg = soft_seg[:, :3 * self.N, :, :]
        masks = torch.cat((input, soft_seg * input, (1 - soft_seg) * input), dim=1)
        hypers = self.Spectral_wise_Transformer(masks)[1]

        return r, hypers


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=[7, 7]):
        super(NCC, self).__init__()
        self.win = win

    def one_dim_ncc(self, y_true, y_pred):
        y = y_true
        x = y_pred


        ndims = len(list(y.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims


        win = [9] * ndims if self.win is None else self.win


        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)


        y2 = y * y
        x2 = x * x
        xy = y * x

        y_sum = F.conv2d(y, sum_filt, stride=stride, padding=padding)
        x_sum = F.conv2d(x, sum_filt, stride=stride, padding=padding)
        y2_sum = F.conv2d(y2, sum_filt, stride=stride, padding=padding)
        x2_sum = F.conv2d(x2, sum_filt, stride=stride, padding=padding)
        xy_sum = F.conv2d(xy, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_y = y_sum / win_size
        u_x = x_sum / win_size

        cross = xy_sum - u_x * y_sum - u_y * x_sum + u_y * u_x * win_size
        y_var = y2_sum - 2 * u_y * y_sum + u_y * u_y * win_size
        x_var = x2_sum - 2 * u_x * x_sum + u_x * u_x * win_size

        cc = (cross * cross / (y_var * x_var + 1e-5)) * (cross / (torch.abs(cross) + 1e-5))
        # cc[cc>0.25] = 0.25
        return -torch.mean(cc)

    def forward(self, y_true, y_pred):
        r_ncc = self.one_dim_ncc(y_true[:, 0:1, :, :], y_pred[:, 0:1, :, :]) / 3.
        g_ncc = self.one_dim_ncc(y_true[:, 1:2, :, :], y_pred[:, 1:2, :, :]) / 3.
        b_ncc = self.one_dim_ncc(y_true[:, 2:3, :, :], y_pred[:, 2:3, :, :]) / 3.

        return g_ncc + b_ncc + r_ncc
