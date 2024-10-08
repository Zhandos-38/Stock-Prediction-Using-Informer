import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.utils import load_yaml, build_object

cfg = load_yaml('config/informer_AAPL_1d.yaml')
experiment_name = cfg['experiment_name']
scale_type = cfg['scale_type']
SAVE_PATH = f'results/{experiment_name}'
os.makedirs(SAVE_PATH, exist_ok=True)


train_dataset = build_object(cfg['train_dataset']['type'], cfg['train_dataset']['module'])(**cfg['train_dataset']['args'])
val_dataset = build_object(cfg['val_dataset']['type'], cfg['val_dataset']['module'])(**cfg['val_dataset']['args'])
test_dataset = build_object(cfg['test_dataset']['type'], cfg['test_dataset']['module'])(**cfg['test_dataset']['args'])

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

train_loader = build_object(cfg['train_loader']['type'], cfg['train_loader']['module'])(train_dataset, **cfg['train_loader']['args'])
val_loader = build_object(cfg['val_loader']['type'], cfg['val_loader']['module'])(val_dataset, **cfg['val_loader']['args'])
test_loader = build_object(cfg['test_loader']['type'], cfg['test_loader']['module'])(test_dataset, **cfg['test_loader']['args'])

model = build_object(cfg['model']['type'], cfg['model']['module'])(**cfg['model']['args'])
criterion = build_object(cfg['criterion']['type'], cfg['criterion']['module'])()
optimizer = build_object(cfg['optimizer']['type'], cfg['optimizer']['module'])(model.parameters(), **cfg['optimizer']['args'])
scheduler = build_object(cfg['scheduler']['type'], cfg['scheduler']['module'])(optimizer, **cfg['scheduler']['args'])

learner = build_object(cfg['learner']['type'], cfg['learner']['module'])(model, criterion, optimizer, **cfg['learner']['args'])
logger = build_object(cfg['logger']['type'], cfg['logger']['module'])(**cfg['logger']['args'])

early_stopping_callback = build_object(cfg['early_stopping']['type'], cfg['early_stopping']['module'])(**cfg['early_stopping']['args'])
model_checkpoint = build_object(cfg['model_checkpoint']['type'], cfg['model_checkpoint']['module'])(**cfg['model_checkpoint']['args'])

trainer = build_object(cfg['trainer']['type'], cfg['trainer']['module'])(
    logger = logger,
    callbacks=[early_stopping_callback, model_checkpoint],
    **cfg['trainer']['args']
)

trainer.fit(learner, train_loader, val_loader)
trainer.test(learner, test_loader)
outputs = trainer.predict(learner, test_loader)
prevs = np.concatenate([out[0] for out in outputs], axis=0)
preds = np.concatenate([out[1] for out in outputs], axis=0)
targets = np.concatenate([out[2] for out in outputs], axis=0)
print(prevs.shape, preds.shape, targets.shape)

np.save(f'{SAVE_PATH}/prevs.npy', prevs)
np.save(f'{SAVE_PATH}/preds.npy', preds)
np.save(f'{SAVE_PATH}/targets.npy', targets)