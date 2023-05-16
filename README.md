# FashionMNIST_Classification

**Binary Image Classification** tasks using **FashionMNIST** dataset, provided by Zalando and containing 60000 training samples of clothing items and additional 10000 samples for testing. 

**Task 1**: Classifying the clothes vs shoe items from the FashionMNIST. Labels 5, 7 and 9 that represent shoe samples are treated as positive (1) while all others, representing shirts, bags or other clothing items are treated as negative (0). 

For this task there are five scripts:
- `data_preprocessing.py`: containts functions for importing, transforming, normalizing and splitting the dataset 
- `model.py`: contains model class, and functions for training and evaluating
- `train.py`: contains the majority of the code for running the training of the model
- `test.py`: evaluates the trained model on the testing samples, and runs the `predict.py` on one random sample
- `predict.py`: plots a sample from the test data, accompanied by ground truth and predicted value by the trained model

`config.json` file contains the parameters for training the model.

The model used for this task is a custom made **Convolutional Neural Network**, with two convolutional layers and one fully-connected layer.
The optimizer chosen is Adam optimizer, and Cross-Entropy loss. Softmax activation function is used to classify the data into two classes, acting as a binary classifier.

The last obtained model parameters are saved in the repository, so the user can run directly the `test.py` script by typing `python test.py` in the terminal, or by running it in the IDE in use.

However, one can run the training again by typing `python train.py` command in the terminal or running it in the IDE. Already existing model parameters will be overwritten in this case.

`unittest` folder contains the `test_data.py` script, which uses the `pytest` testing module. It contains unit tests for data preprocessing, checking mainly If the dataset is of the right format.

**Task 2**: Classifying shoe items from the provided **Shoes** dataset into **Flats** (0) and **Heels** (1). The training set contains only two images of each class respectively, and 20 testing samples, 10 of each class.

The number of training samples is really low, but the pre-trained model from the Task 1 is used to provide the learned features about shoes. This is achieved by freezing the convolutional layers so that during re-training, only classifier (fully-connected layer) gets updated. This technique is known as **transfer learning**. This allows to make the distinction between two different types of shoes.

For this task there is a script:
- `train_eval.py`: re-training, evaluating and predicting

This script is ran by typing `python train_eval.py` command in the terminal, or by running the script in the IDE.

`checkpoint.pt` is a file placed in the parent folder. It is generated as a part of training the model for Task 1. It is saved differently opposed to `best_model.pt` as in addition to the parameters it contains states of epochs, optimizer and losses of the pre-trained model, which are neccessary for the transfer learning performed for Task 2.

The obtained accuracy for Task 1 on the test set, in majority of runs, is ~ 99%. 
The obtained accuracy for Task 2 on the test set, in majority of runs, is ~ 95%.
