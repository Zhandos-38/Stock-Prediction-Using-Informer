import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime

def parse_time(datestr):
    dt = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')
    tt = dt.timetuple()
    return [tt.tm_mon, tt.tm_mday, tt.tm_wday, tt.tm_hour, tt.tm_min]

def create_sequences(df, seq_len, pred_len, target_col='close'):
    datetime = df['datetime'].values
    target_data = df[target_col].values
    data = df.drop(['datetime'], axis=1).values
    inputs, time_inputs, targets, time_targets = [], [], [], []
    
    for i in range(0, len(data) - seq_len - pred_len):
        x = data[i:(i+seq_len)]
        x_mark = [parse_time(str(datetime[j])) for j in range(i, i+seq_len)]
        y = target_data[(i+seq_len):(i+seq_len+pred_len)]
        x_dec_mark = [parse_time(str(datetime[j])) for j in range(i+seq_len, i+seq_len+pred_len)]
        inputs.append(x)
        time_inputs.append(x_mark)
        targets.append(y)
        time_targets.append(x_dec_mark)
        
    return np.array(inputs), np.array(time_inputs), np.array(targets), np.array(time_targets)

class StockDataset(Dataset):
    def __init__(self, csv_path, seq_len, label_len, pred_len, target_col='close'):
        self.df = pd.read_csv(csv_path)
        self.label_len = label_len
        self.pred_len = pred_len
        self.inputs, self.time_inputs, self.targets, self.time_targets = create_sequences(self.df, seq_len, pred_len, target_col)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        x_mark = self.time_inputs[idx]
        y = np.expand_dims(self.targets[idx], -1)
        x_context = x[-self.label_len:, :]
        padding = np.zeros((self.pred_len, x.shape[1]))
        x_dec = np.concatenate([x_context, padding], axis=0)
        x_context_mark = x_mark[-self.label_len:]
        x_padding_mark = self.time_targets[idx]
        x_dec_mark = np.concatenate([x_context_mark, x_padding_mark], axis=0)
        return [torch.tensor(x, dtype=torch.float32),
                torch.tensor(x_mark, dtype=torch.float32),
                torch.tensor(x_dec, dtype=torch.float32),
                torch.tensor(x_dec_mark, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)]