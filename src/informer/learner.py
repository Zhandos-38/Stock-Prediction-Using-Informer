import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from src.utils import destandardize, directional_accuracy
from torchmetrics.functional import mean_absolute_percentage_error

class InformerLearner(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, info_csv, scale_type = 'standard'):
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        df = pd.read_csv(info_csv, index_col=0)
        if scale_type == 'standard':
            self.mean = df['close']['mean']
            self.std = df['close']['std']
            
        if scale_type == 'minmax':
            self.mean = df['close']['min']
            self.std = df['close']['max'] - df['close']['min']
    
    def training_step(self, batch, batch_idx):
        x, x_mark, x_dec, x_mark_dec, y = batch
        pred = self.model(x, x_mark, x_dec, x_mark_dec)
        
        loss = self.criterion(pred, y)
        
        self.log('train_loss', loss.item())
        return {
            'loss': loss
        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out['loss'].detach().cpu() for out in outputs]).mean()
        self.log('train_epoch_loss', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        x, x_mark, x_dec, x_mark_dec, y = batch
        pred = self.model(x, x_mark, x_dec, x_mark_dec)
        
        loss = self.criterion(pred, y)
        mae = nn.L1Loss()(pred, y)
        
        pred_destandardized = destandardize(pred, self.mean, self.std)
        target_destandardized = destandardize(y, self.mean, self.std)
        last_destandardized = destandardize(x[:,-1,0], self.mean, self.std).unsqueeze(1).unsqueeze(1)
        trues, total = directional_accuracy(pred_destandardized, target_destandardized, last_destandardized)
        
        mape = mean_absolute_percentage_error(torch.exp(pred_destandardized), torch.exp(target_destandardized))
        
        return {
            'loss': loss.detach().cpu(),
            'mae': mae.detach().cpu(),
            'trues': trues,
            'total': total,
            'mape': mape.detach().cpu(),
        }
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out['loss'] for out in outputs]).mean()
        avg_mae = torch.stack([out['mae'] for out in outputs]).mean()
        avg_mape = torch.stack([out['mape'] for out in outputs]).mean()
        total_trues = sum([out['trues'] for out in outputs])
        total_total = sum([out['total'] for out in outputs])
        self.log_dict({
            'val_loss': avg_loss,
            'val_mae': avg_mae,
            'val_directional_accuracy': total_trues / total_total,
            'val_mape': avg_mape
        })
    
    def test_step(self, batch, batch_idx):
        x, x_mark, x_dec, x_mark_dec, y = batch
        pred = self.model(x, x_mark, x_dec, x_mark_dec)
        
        loss = self.criterion(pred, y)
        mae = nn.L1Loss()(pred, y)
        
        pred_destandardized = destandardize(pred, self.mean, self.std)
        target_destandardized = destandardize(y, self.mean, self.std)
        last_destandardized = destandardize(x[:,-1,0], self.mean, self.std).unsqueeze(1).unsqueeze(1)
        trues, total = directional_accuracy(pred_destandardized, target_destandardized, last_destandardized)
        
        pred_abs = torch.exp(pred_destandardized)
        target_abs = torch.exp(target_destandardized)
        
        mape = mean_absolute_percentage_error(pred_abs, target_abs)
        abs_mae = nn.L1Loss()(pred_abs, target_abs)
        abs_mse = nn.MSELoss()(pred_abs, target_abs)
        
        return {
            'loss': loss.detach().cpu(),
            'mae': mae.detach().cpu(),
            'trues': trues,
            'total': total,
            'mape': mape.detach().cpu(),
            'abs_mae': abs_mae.detach().cpu(),
            'abs_mse': abs_mse.detach().cpu()
        }
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([out['loss'] for out in outputs]).mean()
        avg_mae = torch.stack([out['mae'] for out in outputs]).mean()
        avg_mape = torch.stack([out['mape'] for out in outputs]).mean()
        abs_mae = torch.stack([out['abs_mae'] for out in outputs]).mean()
        abs_mse = torch.stack([out['abs_mse'] for out in outputs]).mean()
        total_trues = sum([out['trues'] for out in outputs])
        total_total = sum([out['total'] for out in outputs])
        self.log_dict({
            'test_loss': avg_loss,
            'test_mae': avg_mae,
            'test_directional_accuracy': total_trues / total_total,
            'test_mape': avg_mape,
            'test_abs_mae': abs_mae,
            'test_abs_mse': abs_mse
        })
    
    def configure_optimizers(self):
        return self.optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, x_mark, x_dec, x_mark_dec, y = batch
        prev = x[:, :, 0]
        pred = self.model(x, x_mark, x_dec, x_mark_dec)
        
        return prev.cpu().numpy(), pred.cpu().numpy(), y.cpu().numpy()