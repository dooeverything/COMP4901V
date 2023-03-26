import torch
import numpy as np

from models import FCN_ST, SoftmaxCrossEntropyLoss, save_model
from utils import load_dense_data, ConfusionMatrix, DenseCityscapesDataset
import torch.utils.tensorboard as tb
import dense_transforms as dt


def train(args):
    from os import path
    model = FCN_ST()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    
    train_data_path = "datasets/cityscapes/train/"
    valid_data_path = "datasets/cityscapes/valid/"
    
    #trans_data = dense_transforms.Compose([transforms.ToTensor()])
    train_data = DenseCityscapesDataset(train_data_path)
    valid_data = DenseCityscapesDataset(valid_data_path)
    print(f"Train on {train_data.__len__()} samples , Validate on {valid_data.__len__()} samples")
    
    BATCH_SIZE = 32
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"length of the data : {train_data.__len__()}")
    
    #data = train_data.__getitem__(0)    
    #img = dt.label_to_pil_image(data[2])
    #img.show()
    
    EPOCHS = 5
    v_loss_min = 1_000_000
    v_loss = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"run model on {device}")
    model.to(device)
    
    f = open('classweight.cls', 'r')
    loss_weights = []
    for line in f.readlines():
        words = line.split()
        if words[-1] == 'weight':
            continue
        #weight = words[-1].replace(',', '')
        loss_weights.append(float(words[-1]))
            
    #print(loss_weights)
    loss_weights = torch.Tensor(loss_weights).to(device)
    loss_fn = SoftmaxCrossEntropyLoss(loss_weights, ignore_index = 255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    for epoch in range(EPOCHS):
        model.train(True)
        
        epoch_loss = 0.0
        
        loss_list = []
        
        print(f"EPOCH {epoch + 1} for train... with {len(train_loader)}")
        
        for i, data, in enumerate(train_loader):
            X, _, label = data
            X, label = X.to(device), label.to(device)

            pred = model(X)
            #loss = model.loss
            
            optimizer.zero_grad()
            loss = loss_fn(pred, label, device)
            
            #print(f"loss : {loss}")
            if torch.any(torch.isnan(loss)):
                print("error with nan")
                exit()
                
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1)%10 == 0:
                loss_avg = epoch_loss / (i+1)
                loss_list.append(loss_avg)
                #print(f"shape of loss_list {len(loss_list)}")
                print(f"  Batch {i+1} / {len(train_loader)} Loss : {loss_avg}")
                tb_x = epoch * len(train_loader) + i + 1
                train_logger.add_scalar('train/loss', loss_avg, tb_x)
            torch.cuda.empty_cache()

        loss_avg = sum(loss_list) / len(loss_list)
        print(f"  [Train] Loss : {loss_avg}")
        loss_list.clear()
        loss_avg = 0
        
        print(f"EPOCH {epoch+1} for validation...")
        
        v_loss_sum = 0
        with torch.no_grad():
            for i, v_data in enumerate(valid_loader):
                print(f"  {i+1} validation....")
                v_X, _ ,v_label = v_data
                v_X, v_label = v_X.to(device), v_label.to(device)
                v_pred = model(v_X)
                v_loss = loss_fn(v_pred, v_label)
                v_loss_sum += v_loss
        
        v_loss_avg = v_loss_sum / (i+1)
        
        print(f"  [Valid] Loss : {v_loss_avg} \n")
        
        if v_loss_avg < v_loss_min:
            v_loss_min = v_loss_avg
            
        valid_logger.add_scalar('valid/loss', v_loss_min, epoch+1)
        
        torch.cuda.empty_cache()
        train_logger.flush()
        valid_logger.flush()

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
