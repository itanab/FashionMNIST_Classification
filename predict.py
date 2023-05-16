import matplotlib.pyplot as plt
import torch
from model import CNN


def predict_test(test_data, index, model_params):
    """
    :param test_data: data forwarded for testing prediction
    :param index: takes an image from test_data with this index [int]
    :param model_params: path to model params
    :return: plot with ground truth against the predicted value
    """
    with torch.no_grad():

        # Retrieve item
        item = test_data[index]
        image = item[0]
        true_target = item[1]
        input = image[None, :]

        # Loading the saved model
        model = CNN(2)
        model.load_state_dict(torch.load(model_params))
        model.eval()

        # Generate prediction
        prediction = model(input)

        print(prediction)
        pred = torch.argmax(prediction, dim=1)

        # Reshape image
        image = image.reshape(28, 28, 1)

        # Show result
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f'Prediction: {pred} - Actual target: {true_target}')
        plt.show()