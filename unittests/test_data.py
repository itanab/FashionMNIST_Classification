import torch
import data_processing as dp


def test_data_shape():
    train_data, test_data = dp.import_fmnist_data()
    shape_train = train_data[0][0].shape
    shape_test = test_data[0][0].shape
    assert shape_train == torch.Size([1, 28, 28])
    assert shape_test == torch.Size([1, 28, 28])


def test_label_count():
    train_data, _ = dp.import_fmnist_data()
    label_count = len(train_data.targets.unique())
    assert label_count == 10


def test_pixel_range():
    train_data, _ = dp.import_fmnist_data()
    max = torch.max(train_data.data[0])
    min = torch.min(train_data.data[0])
    assert max <= 255
    assert min >= 0


def test_data_count():
    train_data, test_data = dp.import_fmnist_data()
    count_train = len(train_data.data)
    count_test = len(test_data.data)
    assert count_train == 60000
    assert count_test == 10000


def test_post_split_train_data_count():
    train_data, test_data = dp.import_fmnist_data()
    train_data, valid_data = dp.split_train_validate(train_data, 0.9)
    count_train = len(train_data)
    count_val = len(valid_data)
    count_test = len(test_data)
    assert count_train == 54000
    assert count_val == 6000
    assert count_test == 10000


def test_change_labels():
    train_data, test_data = dp.import_fmnist_data()
    dp.change_labels(train_data)
    dp.change_labels(test_data)
    label_count_train = len(train_data.targets.unique())
    label_count_test = len(test_data.targets.unique())
    assert label_count_train == 2
    assert label_count_test == 2

def test_binary_labels():
    train_data, _ = dp.import_fmnist_data()
    dp.change_labels(train_data)
    label_0 = train_data.targets.unique()[0]
    label_1 = train_data.targets.unique()[1]
    assert label_0.item() == 0 and label_1.item() == 1

