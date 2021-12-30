import timm
import torch
from torch import nn
from config import model_name


class MLC(nn.Module):
    """
    自定义分类模型
    """

    def __init__(self, num_classes, device, device_ids=None, pretrained=True):
        super(MLC, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # 改成自己任务的图像类别数
        self.model.fc = nn.Linear(n_features, num_classes, bias=True)
        if device_ids:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
