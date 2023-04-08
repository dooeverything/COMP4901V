import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import resnet50
from dense_transforms import pad_if_smaller
from PIL import Image
import os

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        """
        dropout = 0.1
        
        model_resnet50 = models.resnet50()
        model_load = torch.load("pretrained/resnet50-0676ba61.pth") #models.resnet50(weights="IMAGENET1K_V1")
        model_resnet50.load_state_dict(model_load)

        #self.model = torch.nn.Sequential(*list(model_resnet50.children())[:-1])
        self.resnet_model = torch.nn.Sequential(*list(model_resnet50.children())[:-2])
        self.resnet_model.cuda()

        for parameter in self.resnet_model.parameters():
            parameter.requires_grad = False
        self.resnet_model.eval()
    
        self.pool = nn.AvgPool2d(7)

        # input : (2048, 1, 1) -> output : (1024, 1, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 6)
        )


    def forward(self, x):
        """
        Your code here
        """                
        x = self.resnet_model(x)
        x = self.pool(x)
        x = self.mlp(x)
        x = torch.squeeze(x)
        x = self.head(x)

        return x

class FCN_ST(torch.nn.Module):
    def __init__(self, n_seg):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        self.n_seg = n_seg

        # Load a pretrained resnet50 model
        model_resnet50 = models.resnet50()
        loaded_model = torch.load("pretrained/resnet50-0676ba61.pth") #models.resnet50(weights="IMAGENET1K_V1")        
        model_resnet50.load_state_dict(loaded_model)

        self.encoder = torch.nn.Sequential(*list(model_resnet50.children())[:-2]) 

        self.relu = nn.ReLU()
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding = 1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.upconv2 =  nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.head_seg = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(32, n_seg, kernel_size=1)
        )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """ 
        inputs = x.float()
        x1 = self.encoder[:2](inputs) # [BS, 3, 128, 256] -> [BS, 64, 64, 128]
        x2 = self.encoder[2:5](x1) # [BS, 64, 64, 128] -> [BS, 256, 32, 64]
        x3 = self.encoder[5](x2) #  [BS, 256, 32, 64] -> [BS, 512, 16, 32]
        x4 = self.encoder[6](x3) #  [BS, 512, 16, 32] -> [BS, 1024, 8, 16]
        x5 = self.encoder[7](x4) #  [BS, 1024, 8, 16] -> [BS, 2048, 4, 8]

        x = self.bn1(self.relu(self.upconv1(x5))) 
        x = x + x4 # [BS, 1024, 8, 16]

        x = self.bn2(self.relu(self.upconv2(x)))
        x = x + x3 # [BS, 512, 16, 32]

        x = self.bn3(self.relu(self.upconv3(x)))
        x = x + x2 # [BS, 256, 32, 64]

        x = self.bn4(self.relu(self.upconv4(x))) 
        x = x + x1 # [BS, 64, 64, 128]

        x = self.bn5(self.relu(self.upconv5(x))) # [BS, 32, 128, 256]

        residual = self.conv(inputs)
        x = self.head_seg(x + residual) # [BS, C, 128, 256]
        return x

class FCN_MT(torch.nn.Module):
    def __init__(self, n_seg):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        MODEL_PATH = 'pretrained/fcn_seg.th'
        model_fcn_st =  FCN_ST(n_seg)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model_fcn_st.load_state_dict(checkpoint)
        
        # FCN backbone without the head segmentation
        self.backbone = torch.nn.Sequential(*list(model_fcn_st.children())[:-2])
        print(self.backbone)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.head_seg = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, n_seg, kernel_size=1)
        )

        self.head_depth = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1)
            # nn.Conv2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Conv2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 1, kernel_size=1)
        )

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
        inputs = x.float()
        x = self.backbone(inputs)
        residual = self.conv(inputs)

        x_seg = self.head_seg(x + residual) # [BS, C, 128, 256]
        x_dep = self.head_depth(x) # [BS, 1, 128, 256]

        return x_seg, x_dep

class CrossEntropyLossSeg(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=None):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, labels):
        
        # FCN Loss function
        # input & output : (BS, 19, 128, 256) 
        
        BS, C, H, W = inputs.shape

        inputs = inputs - torch.max(inputs) # prevent the overflow

        exp = torch.exp(inputs)
        sum = torch.sum(exp, dim=1, keepdim=True)

        softmax = exp / (sum + 1e-8)

        # print(f"input : {inputs.shape} or {inputs.min()} or {inputs.max()}")
        # print(f"label : {labels.shape} or {labels.min()} or {labels.max()}")

        if torch.any(torch.isnan(softmax)):
            print(f"softmax : {torch.any(torch.isnan(softmax))} or {softmax.min()} or {softmax.max()}")
            print(sum[0, 0, 0, 0])
            print(f"exp : {torch.any(torch.isnan(exp))} or {exp.min()} or {exp.max()}")
            print(f"sum: {torch.any(torch.isnan(sum))} and {sum[sum==0]} or {sum.min()} or {sum.max()}")
            print(softmax.min())

        # print(f"softmax : {softmax.shape} or {softmax.min()} or {softmax.max()}")

        arr_cat = torch.Tensor([0]).int().cuda()
        weights = torch.cat( (self.weight, arr_cat) )
        labels = torch.unsqueeze(labels, dim=1)
        zeros = torch.zeros((BS, 1, H, W)).cuda()
        softmax = torch.cat((softmax, zeros), dim=1)

        # print(f"softmax after cat: {softmax.shape} or {softmax.min()} or {softmax.max()}")
        # print(f"label : {labels.shape} or {labels.min()} or {labels.max()}")

        labels[labels==self.ignore_index] = C

        # print(f"label after change : {labels.shape} or {labels.min()} or {labels.max()} with {labels.median()}")
        # raise NotImplementedError('stop!')

        loss = torch.gather(softmax, 1, labels)
        loss = torch.squeeze(loss, dim=1) # (BS, 128, 256)
        
        # print(f"loss: {loss.shape} or {loss.min()} or {loss.max()}")

        if torch.any(torch.isnan(loss)):
            print(f"nan with {loss[0].shape}")
            toImg = transforms.ToPILImage()
            img1 = toImg(loss[0])
            img1 = img1
            img1.save('images/img.png')    
            assert not True

        labels = torch.squeeze(labels, dim=1)

        loss = torch.log(loss + 1e-10)
        loss = -weights[labels] * loss

        weights_sum = weights[labels].sum(axis=0)   

        loss = loss.sum(axis=0) / (weights_sum + 1e-10)     

        loss = loss.sum() / (H*W)

        return loss

class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=None):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.weight_list = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """
        # softmax function

        # negative log softmax loss function           
        inputs = inputs - torch.max(inputs) 
        softmax = torch.exp(inputs) /  ( torch.sum(torch.exp(inputs), 1, keepdim=True) + 1e-8)
        loss = -torch.log(softmax[np.arange(softmax.shape[0]), targets]).sum() / inputs.shape[0] 
        return loss

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, depth):

        inputs[inputs<0] = 0
        inputs = torch.log(inputs + 1e-8)
        depth = torch.log(depth + 1e-8)

        depth_ratio = inputs - depth

        N, C, H, W = inputs.shape

        size = C * H * W

        loss1 = torch.sum(depth_ratio * depth_ratio, dim=0) / size
        sum = torch.sum(depth_ratio, dim=0)
        loss2 = (sum*sum) / (2*size*size)

        loss_depth = torch.sum(loss1 - loss2)
        # print(f"loss {loss_depth}")
        return loss_depth

model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST(n_seg=19),
    'fcn_mt': FCN_MT
}

def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join('pretrained/cnn/', path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    print(path.dirname(path.abspath(__file__)))
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r


