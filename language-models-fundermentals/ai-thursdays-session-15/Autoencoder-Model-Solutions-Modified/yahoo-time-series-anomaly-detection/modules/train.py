# train.py

import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import Config
from dataset import YahooTimeSeriesDataset
from model import TimeSeriesAutoencoder
from trainer import Trainer
from evaluator import Evaluator
from plots import plot_losses, plot_roc

cfg = Config()

df = pd.read_csv(cfg.data_path)

values = df["value"].values
labels = df["is_anomaly"].values


n = len(values)

train_end = int(n*0.6)
val_end = int(n*0.9)


train_dataset = YahooTimeSeriesDataset(
    values[:train_end],
    labels[:train_end],
    cfg.sequence_length
)

val_dataset = YahooTimeSeriesDataset(
    values[train_end:val_end],
    labels[train_end:val_end],
    cfg.sequence_length
)

test_dataset = YahooTimeSeriesDataset(
    values[val_end:],
    labels[val_end:],
    cfg.sequence_length
)


train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=cfg.batch_size)
test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size)


model = TimeSeriesAutoencoder(
    cfg.sequence_length,
    cfg.hidden_dim,
    cfg.latent_dim
).to(cfg.device)


optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)

criterion = torch.nn.MSELoss()


trainer = Trainer(model,optimizer,criterion,cfg.device)

trainer.fit(train_loader,val_loader,cfg.epochs)


plot_losses(trainer.train_losses,trainer.val_losses)


evaluator = Evaluator(model,cfg.device)


val_errors,val_labels = evaluator.reconstruction_errors(val_loader)

threshold = evaluator.find_threshold(val_errors)


test_errors,test_labels = evaluator.reconstruction_errors(test_loader)


f1,roc,cm,report,preds = evaluator.evaluate(
    test_errors,
    test_labels,
    threshold
)


print("F1:",f1)
print("ROC:",roc)

print("Confusion Matrix")
print(cm)

print("Classification Report")
print(report)


plot_roc(test_labels,test_errors)