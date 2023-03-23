import os
import torch

import numpy as np

from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

import dense_transforms as dt


class VehicleClassificationDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: load your data from provided dataset (VehicleClassificationDataset) to train your designed model
        """
        # e.g., Bicycle 0, Car 1, Taxi 2, Bus 3, Truck 4, Van 5
        self.data = []
        self.label = {'Bicycle' : 0, 'Car' : 1, 'Taxi' : 2, 'Bus' : 3, 'Truck' : 4, 'Van' : 5}

        for key, value in self.label.items():
            img_path = dataset_path + key
            for img_file in os.listdir(img_path):
                file_name, file_extension = os.path.splitext(img_file)
                if file_extension != '.jpg':
                    print(f"File extension not applicable : {file_name}")
                    continue
                self.data.append([img_path + '/' + img_file, value])

        #raise NotImplementedError('VehicleClassificationDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        #raise NotImplementedError('VehicleClassificationDataset.__len__')
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        Hint: generate samples for training
        Hint: return image, and its image-level class label
        """
        id, label = self.data[idx]
        img = Image.open(id)

        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])

        img_tensor = trans(img)

        return img_tensor, label
        #raise NotImplementedError('VehicleClassificationDataset.__getitem__')

class DenseCityscapesDataset(Dataset):
    def __init__(self, dataset_path, transform=[dt.ToTensor3()]):
        """
        Your code here
        """
        
        print("initiating dataset " + dataset_path)
        
        self.data_path = dataset_path
        self.tran = transform
        self.data_types = ['depth', 'image', 'label']
        
        self.images = []
        self.depths_gt = []
        self.labels_gt = []
        
        for data_type in self.data_types:
            print(data_type)
            data_path = dataset_path + data_type
            for npy_file in os.listdir(data_path):
                npy_path = data_path + '/' + npy_file
                _, file_extension = os.path.splitext(npy_file)
                if file_extension != '.npy':
                    continue
                    
                data = np.load(npy_path, allow_pickle=True)
                
                if data_type == 'depth':
                    self.depths_gt.append(data)
                elif data_type == 'image':
                    self.images.append(data)
                else:
                    self.labels_gt.append(data)
                
                #if data_type == 'depth':
                    #print(data[:,:,0].shape)
                    #print(np.median(data[:,:,0].shape))
                    #im = Image.fromarray((data[:,:,0]*255).astype(np.uint8))
                    #im.show()

        #raise NotImplementedError('DenseCityscapesDataset.__init__')

    def __len__(self):

        """
        Your code here
        """
        return len(self.images)
        #raise NotImplementedError('DenseCityscapesDataset.__len__')

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        #print(f"get item from {idx}")
        image = self.images[idx]
        depth = self.depths_gt[idx]
        label = self.labels_gt[idx]
        
        compose = dt.Compose3(self.tran)
        
        data = compose(image, depth, label)
        
        # im = Image.fromarray((image*255).astype(np.uint8))
        # dp = Image.fromarray((depth[:,:,0]*255).astype(np.uint8))

        # im.show()
        # dp.show()
        
        return data
        
        raise NotImplementedError('DenseCityscapesDataset.__getitem__')


class DenseVisualization():
    def __init__(self, img, depth, segmentation):
        self.img = img
        self.depth = depth
        self.segmentation = segmentation

    def __visualizeitem__(self):
        """
        Your code here
        Hint: you can visualize your model predictions and save them into images. 
        """
        raise NotImplementedError('DenseVisualization.__visualizeitem__')


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = VehicleClassificationDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseCityscapesDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


class DepthError(object):
    def __init__(self, gt, pred):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.gt = gt
        self.pred = pred

    @property
    def compute_errors(self):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((self.gt / self.pred), (self.pred / self.gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        # rmse = (self.gt - self.pred) ** 2
        # rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(self.gt) - np.log(self.pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(self.gt - self.pred) / self.gt)

        # sq_rel = np.mean(((self.gt - self.pred) ** 2) / self.gt)

        return abs_rel, a1, a2, a3
