import torch
import torch.nn as nn
import time
import json

import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

import data_processing as dp
from model import CNN, train, evaluate, epoch_time


with open('config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("{} in usage!".format(device))

#importing and transforming data sets
train_data, _ = dp.import_fmnist_data()
dp.change_labels(train_data)
train_data, valid_data = dp.split_train_validate(train_data, config["valid_ratio"])

#creating data loaders/iterators
train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=config["batch_size"])

valid_iterator = data.DataLoader(valid_data,
                                 batch_size=config["batch_size"])

#creating model instances
model = CNN(config["output_dim"])
print(model)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

#training
best_valid_loss = float('inf')
train_losses, valid_losses = [], []

for epoch in range(config["epochs"]):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, 'checkpoint.pt')

    end_time = time.monotonic()

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


#plotting losses
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.legend()
plt.savefig('./plots/loss.png')

