import torch
import torch.nn as nn

class TrendNormalize(nn.Module):
    def __init__(self):
        super(TrendNormalize, self).__init__()

    def forward(self, x):
        """
        对数据进行趋势化
        :param x: 待处理的数据，PyTorch张量
        :return: 趋势化后的数据，PyTorch张量
        """
        # 计算数据的均值和标准差
        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True)
        # 对数据进行标准化
        x = (x - mean) / std
        # 计算数据的趋势
        trend = torch.zeros_like(x)
        trend[1:] = x[1:] - x[:-1]
        # 对趋势进行标准化
        trend_mean = torch.mean(trend, dim=0, keepdim=True)
        trend_std = torch.std(trend, dim=0, keepdim=True)
        trend = (trend - trend_mean) / trend_std
        # 将趋势化后的数据和趋势合并
        data_trend = torch.zeros_like(x)
        data_trend[0] = x[0]
        data_trend[1:] = data_trend[:-1] + trend
        return data_trend