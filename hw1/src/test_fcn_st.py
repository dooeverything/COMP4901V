import torch
import numpy as np

from models import FCN_ST 
from utils import load_dense_data, ConfusionMatrix, DenseCityscapesDataset
import dense_transforms as dt
from torch.utils.data import DataLoader 
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your single-task model, and perform evaluation for the segmentation task
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task

    """
    model = FCN_ST(n_seg=19)
    path = 'pretrained/fcn_seg.th'
    model.load_state_dict(torch.load(path))
    model.cuda()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    transform = dt.Compose([
        dt.ToTensor(),
    ])
    data_path = 'datasets/cityscapes/test/'
    test = DenseCityscapesDataset(data_path, transform)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    confusion = ConfusionMatrix(size=19)
    accuracy = 0.0
    mIoU = 0.0
    print(f"Test on {test.__len__()} samples")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, label, _ = data
            X, label = X.cuda(), label.cuda()

            pred = model(X)
            pred_max = pred.argmax(1)
            confusion.add(pred_max, label)
            accuracy += confusion.global_accuracy
            mIoU += confusion.iou


    accuracy = accuracy/(i+1)
    mIoU = mIoU/(i+1)

    print(f"****Segmentation : Accuracy : {(accuracy*100):.2f}%, iou : {mIoU:.4f}****")
    return accuracy, mIoU


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
