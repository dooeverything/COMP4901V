import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

import numpy as np

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        """

        model_resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        #self.model = torch.nn.Sequential(*list(model_resnet50.children())[:-1])
        self.model = torch.nn.Sequential(*list(model_resnet50.children())[:-2])
        
        self.avgPooling = nn.AvgPool2d(7)
        
        self.linear = nn.Linear(2048, 1024)

        self.mlp_head = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Conv2d(2048, 1024, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 6, 1)
        )

        self.loss = 0

        #raise NotImplementedError('CNNClassifier.__init__') 

    def forward(self, x):
        """
        Your code here
        """        
        batch_size = x.shape[0]
        # print(f"Batch size : {batch_size}")
        x = self.model(x)
        #print(f"Shape of output of resnet: {x.shape}")

        x = self.mlp_head(x)
        #print(f"Shape of output of mlp head: {x.shape}")

        x = torch.reshape(x, (batch_size, 6)) # convert (batchsize, number of class to detect, 1, 1) to (batchsize, number of class to detect)
        #print(f"Shape of output of reshape: {x.shape}")

        return x
        #raise NotImplementedError('CNNClassifier.forward') 


class FCN_ST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """



        raise NotImplementedError('FCN_ST.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        raise NotImplementedError('FCN_ST.forward')


class FCN_MT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_MT.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN_MT.forward')


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):

        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """

        #print("inputs in loss function : ")
        #print(inputs[0])

        y = torch.exp(inputs) / torch.sum(torch.exp(inputs)) # softmax function
        
        #print("y in loss function: ")
        #print(y[0])
        
        #print("target in loss function : ")
        #print(targets[0])

        loss = -torch.sum(torch.log(y[targets])) # negative log softmax loss function

        return loss

        #raise NotImplementedError('SoftmaxCrossEntropyLoss.__init__')


model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST,
    'fcn_mt': FCN_MT
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
