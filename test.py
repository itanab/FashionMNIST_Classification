import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import data_processing
from model import CNN, evaluate
from predict import predict_test


with open('config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("{} in usage!".format(device))

# Importing and transforming test data
_, test_data = data_processing.import_fmnist_data()
data_processing.change_labels(test_data)

test_iterator = data.DataLoader(test_data, batch_size=config["batch_size"])

# Creating model instances
model = CNN(config["output_dim"])
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

# Evaluating the model
try:
    model.load_state_dict(torch.load('best_model.pt'))
except OSError:
    print("Model does not exist!")
    sys.exit()

test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

predict_test(test_data, np.random.randint(0, len(test_data)), 'best_model.pt')