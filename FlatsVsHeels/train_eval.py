import torch
import torch.nn as nn
import time
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

from model import CNN, train, evaluate, epoch_time
from predict import predict_test

with open('../config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("{} in usage!".format(device))

# Importing and transforming data sets
transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

train_data = datasets.ImageFolder("shoes/train/", transform=transformations)
test_data = datasets.ImageFolder("shoes/test/", transform=transformations)

# Creating data loaders/iterators
train_iterator = data.DataLoader(train_data, shuffle=True)
test_iterator = data.DataLoader(test_data)

# Creating model instances
model = CNN(config["output_dim"])
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

checkpoint = torch.load('../checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']

# Freezing weights and biases of convolution layers
model.conv1.weight.requires_grad = False
model.conv1.bias.requires_grad = False

model.conv2.weight.requires_grad = False
model.conv2.bias.requires_grad = False

model = model.to(device)
criterion = criterion.to(device)

# Resuming training with frozen layers and new data
train_losses = []
best_train_loss = float('inf')

for epoch in range(20):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), 'best_model_t2.pt')

    end_time = time.monotonic()

    train_losses.append(train_loss)

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

# Plotting losses
plt.plot(train_losses, label='Train Loss')
plt.legend()
plt.savefig('../plots/loss_task2.png')

print('----------------- TESTING-----------------')
model.load_state_dict(torch.load('best_model_t2.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

predict_test(test_data, np.random.randint(1, len(test_data)), 'best_model_t2.pt')