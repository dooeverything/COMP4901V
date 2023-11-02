from .models import CNNClassifier
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
from torch.utils.data import DataLoader 
import torchvision
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    model = CNNClassifier()

    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for the vehicle classification task
    Hint: use the ConfusionMatrix for you to calculate accuracy
    """

    model = CNNClassifier()
    path = 'pretrained/cnn.th'
    model.load_state_dict(torch.load(path))
    model.cuda()

    data_path = 'datasets/vehicle/test'
    test_data = VehicleClassificationDataset(dataset_path=data_path, train=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    confusion = ConfusionMatrix(size=6)
    test_acc = 0
    model.train(False)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, t = data
            X, t = X.cuda(), t.cuda()

            pred = model(X)
            pred_max = pred.argmax(1)
            confusion.add(pred_max, t)
            test_acc += confusion.global_accuracy
    
    accuracy = test_acc / (i+1)

    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
