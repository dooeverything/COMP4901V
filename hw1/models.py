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
        dropout = 0.1
        model_resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        #self.model = torch.nn.Sequential(*list(model_resnet50.children())[:-1])
        self.model = torch.nn.Sequential(*list(model_resnet50.children())[:-2])
                
        self.linear = nn.Linear(2048, 1024)

        self.avg = nn.AvgPool2d(7)
        
        self.mlp_head_1 = nn.Sequential(
            #nn.Linear(2048, 1024), 
            nn.Conv2d(2048, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout(dropout),
        )
        
        self.mlp_head_2 = nn.Sequential(
            nn.Linear(1024, 6),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.loss = 0

        #raise NotImplementedError('CNNClassifier.__init__') 

    def forward(self, x):
        """
        Your code here
        """        
        batch_size = x.shape[0]
        #print(f"Batch size : {batch_size}")
        
        x = self.model(x)
        #print(f"Shape of output of resnet: {x.shape}")

        x = self.avg(x)
        x = self.mlp_head_1(x)
        
        x = torch.reshape(x, (batch_size, 1024))
        x = self.mlp_head_2(x)
        #print(f"Shape of output of mlp head: {x.shape}")

        #print(f"Shape of output of mlp head: {x.shape}")

        #x = torch.reshape(x, (batch_size, 6)) # convert (batchsize, number of class to detect, 1, 1) to (batchsize, number of class to detect)
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
        dropout = 0.1
        model_resnet50 = models.resnet50(weights="IMAGENET1K_V1") # output : (BS, 1024, 1)
        
        # Implement FCN - Fully Connected Networks for Image Segmentations
        self.encoder = torch.nn.Sequential(*list(model_resnet50.children())[:-2]) # output : (BS, 2048, 7, 7)
        
        #print(self.encoder)
        
        self.neck = torch.nn.Sequential(
            #nn.AvgPool2d(7), # output : (BS, 2048, 7, 7)
            nn.Conv2d(2048, 1024, 1), # output : (BS, 1024, 7, 7)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(),
        ) # output : (BS, 1024, 7, 7)
        
        # input : (BS, 1024, 7, 7)
        self.decoder1 = torch.nn.Sequential(
            #nn.ConvTranspose2d(1024, 2048, 1), #input : (BS, 1024, 7, 7) -> (BS, 2048, 7, 7)
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding = 1, output_padding=1), #input : (BS, 1024, 7, 7) -> (BS, 512, 14, 14)
            nn.ConvTranspose2d(512, 512, kernel_size=1),
        )
        
                    
        #input : (BS, 512, 14, 14) -> (BS, 256, 28, 28)
        self.decoder2 = torch.nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=1),
        )
        
        # input : (BS, 256, 28, 28) -> (BS, 128, 56, 56)
        self.decoder3 = torch.nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=1)
        )
        
        # input : (BS, 128, 56, 56) -> (BS, 64, 112, 112)
        self.decoder4 = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 64, kernel_size=1),
        )
        
        # input : (BS, 64, 112, 112) -> (BS, 32, 224, 335)
        self.decoder5 = torch.nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,3), padding=1, output_padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=1),
        )
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding = 1, output_padding=1), #input : (BS, 1024, 7, 7) -> (BS, 512, 14, 14)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,3), padding=1, output_padding=1),
        )
        
        # input : (BS, 19, 224, 224)
        self.head = torch.nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 19, kernel_size=1)
        )
        
        #raise NotImplementedError('FCN_ST.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        batch_size = x.shape[0]
        
        #print(f"the shape of the original input : {x.shape}")
        
        batch_size, channel, H, W = x.shape
                
        resize_input = transforms.Compose([transforms.Resize((224, 224))])
        
        x = resize_input(x).float()
        
        #print(f"the shape of the input: {x.shape}")
        
        x = self.encoder(x)
        
        x = self.neck(x)
        
        # x = self.decoder1(x)
        # x = self.decoder2(x)
        # x = self.decoder3(x)
        # x = self.decoder4(x)
        # x = self.decoder5(x)
        
        x = self.decoder(x)
        #print(f"the shape of the input: {x.shape}")
        
        x = self.head(x)
        #print(f"the size of x after head : {x.shape}")
        
        x = x[:, :, :H, :W]
        #print(f"the final out of x : {x.shape}")
        
        return x
        #raise NotImplementedError('FCN_ST.forward')


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
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.weight_list = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, device=None):

        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """

        #print(inputs[0])
        #batch_size = inputs.shape[0]
        
        #print("y in loss function: ")
        #print(y[0])
        
        #print(f"target in loss function : {targets.shape}")
        #print(targets[0])
        #print(f"inputs in loss function : {inputs.shape} and {targets.shape}")

        softmax = 0
        #sum_softmax = torch.sum(torch.exp(inputs), 1, keepdim=True)
        #print(f"sum_softmax shape {sum_softmax.shape}")
        if self.ignore_index is None:
            softmax = torch.exp(inputs) /  torch.sum(torch.exp(inputs), 1, keepdim=True) # softmax function
        else:
            # FCN Loss function
            # input & output : (BS, 19, 128, 256) 
            inputs_exp = torch.exp(inputs)
            inputs_exp_sum = inputs_exp.sum(axis=1, keepdim=True)
            softmax = inputs_exp / inputs_exp_sum
            #print(f"softmax function : {inputs_exp.shape} / {inputs_exp_sum.shape} = {softmax.shape}")
        
        #softmax_test = F.softmax(inputs)
        #print(f"softmax vs softmax_test {softmax.shape} vs {softmax_test.shape}")
        #print(f"softmax with targets column : {softmax[np.arange(softmax.shape[0]), targets]}")
        #print(f"softmax with targets column : {-torch.log(softmax[np.arange(softmax.shape[0]), targets])}")
        #loss_test = F.cross_entropy(inputs, targets)
        
        # negative log softmax loss function
        if self.weight_list is None:
            loss = -torch.log(softmax[np.arange(softmax.shape[0]), targets]).sum() / inputs.shape[0] 
        else:
            #weights = self.weight_list[targets].reshape((BS, H*W))
            targets_flatten = torch.flatten(targets, 1, 2)
            softmax_flatten = torch.flatten(softmax, 2, 3)
            #print(f"loss function : {targets_flatten.shape} and {softmax_flatten.shape} ")
            BS, C, H_W = softmax_flatten.shape
            
            softmax_transpose = softmax_flatten.mT
            targets_transpose = targets_flatten.mT
            
            #print(f"soft : {softmax_transpose.shape} and targets_bs : {targets_transpose.shape}")
            softmax_ts = []
            for bs in range(BS):
                softmax_bs = softmax_transpose[bs, :, :]
                targets_bs = targets_transpose[:, bs]
                #print(f"soft : {softmax_bs.shape} and targets_bs : {targets_bs.shape}")
                weights = self.weight_list[targets_bs]
                #print(f"weights: {weights.shape}")
                softmax_t = -weights * torch.log(softmax_bs[torch.arange(H_W), targets_bs])
                softmax_t = softmax_t.tolist()
                softmax_ts.append(softmax_t)
                #print(f"test : {len(test)}")
                
            softmax_ts = torch.Tensor(softmax_ts).to(device)
            softmax_ts = softmax_ts.reshape((BS, inputs.shape[2], inputs.shape[3]))
            
            #print(f"softmax_t : {softmax_ts.shape}")
            #loss = F.cross_entropy(inputs, targets, reduction='none')
            #loss = -weights * torch.log(softmax_t).sum(axis=1, keepdim=True) / inputs.shape[0]
        
            loss = softmax_ts.sum(axis=0) / BS
            
            loss.requires_grad_(True)
            
            loss = loss.sum() / H_W
            
            #print(f"Final Loss: {loss.shape} ") # (H, W)
        #print(f"loss vs loss+test : {loss} vs {loss_test}")
        
        #raise NotImplementedError('SoftmaxCrossEntropyLoss.__init__')
        return loss



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
