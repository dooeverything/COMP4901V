import torch
import numpy as np

from models import FCN_MT, CrossEntropyLossSeg, save_model, DepthLoss
from utils import load_dense_data, ConfusionMatrix, DenseCityscapesDataset
import torch.utils.tensorboard as tb
import dense_transforms as dt
from torchvision import transforms
import torch.nn as nn
from datetime import datetime
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        PATH = 'FCN_MT_{}'.format(timestamp)
        train_logger = tb.SummaryWriter(path.join(args.log_dir, PATH ,'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, PATH ,'valid'), flush_secs=1)

    """
    Your code here
    Hint: validation during training: use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too. 
    Hint: Use the log function below to debug and visualize your model
    """

    train_data_path = "datasets/cityscapes/train/"
    valid_data_path = "datasets/cityscapes/valid/"

    transform = dt.Compose([
                            dt.RandomHorizontalFlip(),
                            dt.ColorJitter(),
                            dt.ToTensor()
                            ])

# dt.RandomHorizontalFlip(),
# dt.ColorJitter(),

    BATCH_SIZE = 16
    EPOCHS = 120

    train_data = DenseCityscapesDataset(train_data_path, transform=transform)
    valid_data = DenseCityscapesDataset(valid_data_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"run model on {device}")
    
    model = FCN_MT(n_seg=19)
    model.backbone.requires_grad_ = False
    model.cuda()
    
    f = open('classweight.cls', 'r')
    loss_weights = []
    for line in f.readlines():
        words = line.split()
        if words[-1] == 'weight':
            continue
        loss_weights.append(float(words[-1]))

    loss_weights = torch.Tensor(loss_weights).cuda()
    loss_seg = CrossEntropyLossSeg(weight=loss_weights, ignore_index=255)
    loss_depth = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=1e-6)

    v_loss_min_seg = 1_000_000
    v_loss_min_depth = 1_000_000
    v_loss_seg_prev = 0
    v_loss_depth_prev = 0
    patience = 5
    counter = 0
    model.train()
    print(f"Train on {train_data.__len__()} samples, Validate on {valid_data.__len__()} samples")
    for epoch in range(EPOCHS):
        print(f"EPOCH [{epoch + 1}/{EPOCHS}] for train... with {len(train_loader)}")        
        
        loss_sum_seg = 0
        loss_avg_seg = 0

        loss_sum_depth = 0
        loss_sum_depth = 0

        for i, data, in enumerate(train_loader):
            X, label, depth = data
            X, label, depth = X.cuda(), label.cuda(), depth.cuda()

            pred_seg, pred_depth = model(X)
            optimizer.zero_grad()
            loss1 = loss_seg(pred_seg, label)
            loss2 = loss_depth(pred_depth, depth)

            loss = loss1 + loss2

            if torch.isnan(loss1) or torch.isnan(loss2):
                print("error with nan")
                assert not True

            loss.backward()

            optimizer.step()
            loss_sum_seg += loss1.item()
            loss_sum_depth += loss2.item()
                        
            if (i+1)%10 == 0:
                loss_avg_seg = loss_sum_seg / (i+1)
                loss_avg_depth = loss_sum_depth / (i+1)

                tb_x = epoch * len(train_loader) + i + 1
                depth = torch.squeeze(depth, dim=1)
                pred_depth = torch.squeeze(pred_depth, dim=1)
                train_logger.add_scalar('train/loss_seg', loss_avg_seg, tb_x)
                train_logger.add_scalar('train/loss_depth', loss_avg_depth, tb_x)

                print(f"  [Batch {i+1}/{len(train_loader)}] Loss for seg : {loss_avg_seg:.4f}, Loss for depth : {loss_avg_depth:.6f}")

        loss_avg_seg = loss_sum_seg / (i+1)
        loss_avg_depth = loss_sum_depth / (i+1)

        print(f"  [Train] Loss for seg : {loss_avg_seg:.4f}, Loss for depth : {loss_avg_depth:.6f}")

        v_loss_sum_seg = 0.0
        v_loss_sum_depth = 0.0

        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_loader):
                v_X, v_s, v_d = v_data
                v_X, v_s, v_d = v_X.cuda(), v_s.cuda(), v_d.cuda()

                v_pred1, v_pred2 = model(v_X)
                v_loss1 = loss_seg(v_pred1, v_s)
                v_loss_sum_seg += v_loss1

                v_loss2 = loss_depth(v_pred2, v_d)
                v_loss_sum_depth += v_loss2
                v_d = torch.squeeze(v_d)

        v_loss_avg_seg = v_loss_sum_seg / (i+1)
        v_loss_avg_depth = v_loss_sum_depth / (i+1)

        tb_v = epoch * len(valid_loader) + i + 1
        print(f"  [Valid] Loss for seg : {v_loss_avg_seg:.4f}, Loss for depth : {v_loss_avg_depth:.6f} \n")
        
        valid_logger.add_scalar('valid/loss_seg', v_loss_avg_seg, tb_v)
        valid_logger.add_scalar('valid/loss_depth', v_loss_avg_depth, tb_v)
        
        # scheduler.step()
        train_logger.flush()
        valid_logger.flush()

        if v_loss_avg_seg + 1e-5 < v_loss_seg_prev and v_loss_avg_depth + 1 < v_loss_depth_prev:
            print(f"save checkpoint....\n") # iou best Update! v_iou_best : {v_iou_best} vs v_iou_avg : {v_iou_avg} 
            counter = 0
            torch.save(model.state_dict(), 'pretrained/fcn_mt.th')
        else:
            counter += 1
            print(f"[counter/patience] : [{counter}/{patience}]... \n")

        if (epoch+1)%5==0:
            print(f"aave depth image at {epoch+1}.... \n")
            save(X, depth, label, pred_depth, pred_seg, epoch+1, 'train/')
            save(v_X, v_d, v_s, v_pred2, v_pred1, epoch+1, 'valid/')
        
        v_loss_seg_prev = v_loss_avg_seg
        v_loss_depth_prev = v_loss_avg_depth


        if patience < counter:
            print(f"Number of counter exceeds patience... early stopping")
            break
    print("Finish training...")

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', 
                    np.array(dt.label_to_pil_image(lbls[0].cpu()).convert('RGB')), 
                    global_step, dataformats='HWC')
    
    logger.add_image('prediction', 
                    np.array(dt.label_to_pil_image(logits[0].argmax(dim=0).cpu()).convert('RGB')),
                    global_step, dataformats='HWC')


def save(img, depth, label, logits1, logits2, global_step, where):
    idx = torch.randint(0, depth.shape[0], size=(1,))
    
    save_image(img[idx],fp='images/image/' + where  + str(global_step)+'.png' )
    save_image(depth[idx], fp='images/depth_gt/'+ where  + str(global_step)+'.png')
    save_image(logits1[idx], fp='images/depth_pred/' + where  + str(global_step)+'.png')
    
    im = dt.label_to_pil_image(logits2[0].argmax(dim=0).cpu()).convert('RGB')
    im.save(fp='images/seg/'+ where  + str(global_step)+'.png' )
    lb = dt.label_to_pil_image(label[0].cpu()).convert('RGB')
    lb.save(fp='images/lb/'+ where  + str(global_step)+'.png' )




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
