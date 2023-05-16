import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data


def calculate_mean_std():
    """
    Calculates the mean and the standard deviation of the training dataset
    :return: mean and std
    """
    train_set = datasets.FashionMNIST(root="./",
                                      download=True,
                                      train=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))

    loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
    data = next(iter(loader))

    mean = data[0].mean()
    std = data[0].std()

    return mean, std


mean, std = calculate_mean_std()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def import_fmnist_data():
    """
    :return: imported FashionMNIST training and test dataset
    """
    train_data = datasets.FashionMNIST(root='./',
                                        train=True,
                                        download=True,
                                        transform=transform)

    test_data = datasets.FashionMNIST(root='./',
                                        train=False,
                                        download=True,
                                        transform=transform)
    return train_data, test_data


def change_labels(data_set):
    """
    Changes the labels to fit the binary classification task
    :return: does the changes in place
    """
    for i, val in enumerate(data_set.targets):
        if val == 5 or val == 7 or val == 9:
            data_set.targets[i] = 1
        else:
            data_set.targets[i] = 0


def split_train_validate(train_data, ratio):
    """
    :param ratio: ratio of images taken for training data set
    :return: split training and validation data set
    """
    n_train_examples = int(len(train_data) * ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

    return train_data, valid_data








