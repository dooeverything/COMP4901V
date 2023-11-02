import torch
import numpy as np

from models import FCN_MT
from utils import ConfusionMatrix, DepthError, DenseCityscapesDataset, Dense
import dense_transforms as dt
from torch.utils.data import DataLoader 
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """

    model = FCN_MT(n_seg=19)
    path = 'pretrained/fcn_mt.th'
    model.load_state_dict(torch.load(path))
    model.cuda()


    print(model)
    data_path = 'datasets/cityscapes/test/'

    transform = dt.Compose([
        dt.ToTensor(),
    ])

    test = DenseCityscapesDataset(data_path, transform)
    test_loader = DataLoader(test)

    confusion = ConfusionMatrix(size=19)
    accuracy = 0.0
    mIoU = 0.0
    rel_sum, a1_sum, a2_sum, a3_sum = 0.0, 0.0, 0.0, 0.0
    print(f"Test on {test.__len__()} samples")
    
    model.train(False)
    model.eval()

    n_vis = 6
    random_indices = torch.randperm(test.__len__())[:n_vis]
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, label, depth = data
            X, label, depth = X.cuda(), label.cuda(), depth.cuda()

            pred1, pred2 = model(X)
            pred_max = pred1.argmax(1)
            confusion.add(pred_max, label)
            accuracy += confusion.global_accuracy
            mIoU += confusion.iou

            rel, a1, a2, a3 = DepthError(depth, pred2).compute_errors
            rel_sum += rel
            a1_sum += a1
            a2_sum += a2
            a3_sum += a3

            if i in random_indices:
                vis_gt = Dense(img=X, depth=[depth, pred2], segmentation=[label, pred1])
                vis_gt.__visualizeitem__(idx=i+1)


    accuracy = accuracy/(i+1)
    mIoU = mIoU/(i+1)
    rel = rel_sum/(i+1)
    a1 = a1_sum/(i+1)
    a2 = a2_sum/(i+1)
    a3 = a3_sum/(i+1)

    print(f"****Segmentation : Accuracy : {(accuracy*100):.2f}%, iou : {mIoU:.4f}****")
    print(f"****Depth : rel : {rel:.4f}%, a1 : {a1:.4f}, a2 : {a2:.4f}, , a3 : {a3:.4f}****")
    
    return accuracy, mIoU, rel, a1, a2, a3


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    args = parser.parse_args()
    test(args)
