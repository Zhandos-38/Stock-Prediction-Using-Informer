import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
    
class MAPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return torch.mean(torch.abs((y - yhat) / y)) * 100