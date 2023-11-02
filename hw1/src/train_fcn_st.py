import torch
import numpy as np

from models import FCN_ST, CrossEntropyLossSeg, save_model
from utils import load_dense_data, ConfusionMatrix, DenseCityscapesDataset
import torch.utils.tensorboard as tb
import dense_transforms as dt
from torchvision import transforms
import torch.nn as nn
from datetime import datetime
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split

def train(args):
    from os import path
    train_logger, valid_logger = None, None

    if args.bs is None:
        args.bs = 16

    if args.lr is None:
        args.lr = '1e-4'

    if args.wd is None:
        args.wd = '5e-5'

    if args.freeze is None:
        args.freeze = False
    
    if args.custom is None:
        args.custom = 'True'

    if args.logdir is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        PATH = 'FCN_ST_{}_bs:{}_lr:{}_weight_decay:{}_freeze:{}_custom:{}'.format(timestamp, args.bs, args.lr, args.wd, args.freeze, args.custom)
        train_logger = tb.SummaryWriter(path.join(args.logdir, PATH ,'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.logdir, PATH ,'valid'), flush_secs=1)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    
    train_data_path = "datasets/cityscapes/train/"
    valid_data_path = "datasets/cityscapes/valid/"
    transform_train = dt.Compose([
        dt.RandomHorizontalFlip(),
        dt.ToTensor(),
    ])

    transform_valid = dt.Compose([
        dt.RandomHorizontalFlip(),
        dt.ToTensor(),
    ])

    BATCH_SIZE = int(args.bs)
    EPOCHS = 100

    train = DenseCityscapesDataset(train_data_path, transform_train)
    valid = DenseCityscapesDataset(valid_data_path, transform_valid)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"run model on {device}")
    
    fcn_model = FCN_ST(n_seg=19)
    fcn_model.cuda()
    # Freeze the parameter in encoder
    for parameter in fcn_model.encoder.parameters():
        if args.freeze:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # Load class weights for loss function     
    f = open('classweight.cls', 'r')
    loss_weights = []
    for line in f.readlines():
        words = line.split()
        if words[-1] == 'weight':
            continue
        loss_weights.append(float(words[-1]))

    loss_weights = torch.Tensor(loss_weights).cuda()

    if args.custom == 'True':
        print(f"use custom loss : {args.custom}")
        loss_fn = CrossEntropyLossSeg(weight=loss_weights, ignore_index=255)
    else:
        print(f"use nn loss : {args.custom}")
        loss_fn = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
    
    optimizer = torch.optim.RMSprop(fcn_model.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=float(args.wd))    

    patience = 5
    v_loss_min = 1_000_000
    counter = 0

    print(f"Train on {train.__len__()} samples, Validate on {valid.__len__()} samples")
    fcn_model.train()
    for epoch in range(EPOCHS):
        print(f"EPOCH [{epoch + 1}/{EPOCHS}] for train... with {len(train_loader)}")        
        
        loss_sum = 0
        loss_avg = 0
        acc_sum = 0
        acc_avg = 0
        iou_sum = 0
        iou_avg = 0

        for i, data, in enumerate(train_loader):
            X, label, _ = data
            X, label  = X.cuda(), label.cuda()
            
            if X.dim() == 5:
                BS, n, C, H, W = X.shape
                X = torch.reshape(X, (BS*n, C, H, W))

            if label.dim() == 4:
                BS, n, H, W = label.shape
                label = torch.reshape(label, (BS*n, H, W))

            pred = fcn_model(X)
            optimizer.zero_grad()
            loss = loss_fn(pred, label)

            if torch.isnan(loss):
                print("error with nan")
                exit()
            
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
            pred_max = pred.argmax(1)
            confusion = ConfusionMatrix(size=19)
            confusion.add(pred_max, label)
            acc = confusion.global_accuracy
            iou = confusion.iou

            acc_sum += acc
            iou_sum += iou
            
            if (i+1)%100 == 0:
                loss_avg = loss_sum / (i+1)
                acc_avg = acc_sum / (i+1)
                iou_avg = iou_sum / (i+1)
                tb_x = epoch * len(train_loader) + i + 1
                
                log(train_logger, X, label, pred, tb_x)
                train_logger.add_scalar('train/loss', loss_avg, tb_x)
                train_logger.add_scalar('train/acc', acc_avg, tb_x)
                train_logger.add_scalar('train/iou', iou_avg, tb_x)
                torch.cuda.empty_cache()

                print(f"  [Batch {i+1}/{len(train_loader)}] Loss : {loss_avg:.4f}, Accuracy : {(acc_avg*100):.2f}%, iou : {iou_avg:.4f}")

        loss_avg = loss_sum / (i+1)
        acc_avg = acc_sum / (i+1)
        iou_avg = iou_sum / (i+1)

        print(f"  [Train] Loss avg : {loss_avg:.4f}, Acc avg : {(acc_avg*100):.2f}%, iou avg : {iou_avg:.4f}")

        v_loss_sum = 0.0
        v_acc_sum = 0.0
        v_iou_sum = 0.0
        v_confusion = ConfusionMatrix(size=19)
        fcn_model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_loader):
                v_X, v_t, _ = v_data
                v_X, v_t = v_X.cuda(), v_t.cuda()

                if v_X.dim() == 5:
                    BS, n, C, H, W = v_X.shape
                    v_X = torch.reshape(v_X, (BS*n, C, H, W))

                if v_t.dim() == 4:
                    BS, n, H, W = v_t.shape
                    v_t = torch.reshape(v_t, (BS*n, H, W))

                v_pred = fcn_model(v_X)
                v_loss = loss_fn(v_pred, v_t)
                v_loss_sum += v_loss

                v_pred_max = v_pred.argmax(1)
                v_confusion.add(v_pred_max, v_t)
                v_acc_sum += v_confusion.global_accuracy
                v_iou_sum += v_confusion.iou

                v_loss_avg = v_loss_sum / (i+1)
                v_acc_avg = v_acc_sum / (i+1)
                v_iou_avg = v_iou_sum / (i+1)
                tb_v = epoch * len(valid_loader) + i + 1

                log(valid_logger, v_X, v_t, v_pred, tb_v)
                valid_logger.add_scalar('valid/loss', v_loss_avg, tb_v)
                valid_logger.add_scalar('valid/acc', v_acc_avg, tb_v)
                valid_logger.add_scalar('valid/iou', v_iou_avg, tb_v)


        print(f"  [Valid] Loss : {v_loss_avg:.4f}, Acc avg : {(v_acc_avg * 100):.2f}%, iou avg : {v_iou_avg:.4f} \n")
                    
        train_logger.flush()
        valid_logger.flush()

        if  v_loss_avg < v_loss_min + 5e-4:
            v_loss_min = v_loss_avg
            counter = 0
            print(f"save checkpoint....\n") 
            torch.save(fcn_model.state_dict(), 'pretrained/fcn_seg_' + timestamp + '.th')
        elif v_loss_avg >= v_loss_min:
            counter += 1
            print(f"[counter/patience] : [{counter}/{patience}]... \n")

        if counter > patience:
            print(f"Number of counter exceeds patience... early stopping")
            break   
    
    print(f"Finish Training....")



def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dt.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dt.label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir')
    # Put custom arguments here
    parser.add_argument('--bs')
    parser.add_argument('--lr')
    parser.add_argument('--wd')
    parser.add_argument('--freeze')
    parser.add_argument('--custom')

    args = parser.parse_args()
    train(args)
