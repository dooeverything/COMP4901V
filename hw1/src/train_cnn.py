from models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from datetime import datetime
from dense_transforms import RandomHorizontalFlip, ColorJitter
from torchvision import transforms
import torch.nn.functional as F

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.logdir is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        PATH = 'CNN_{}'.format(timestamp)
        train_logger = tb.SummaryWriter(path.join(args.logdir, PATH ,'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.logdir, PATH ,'valid'), flush_secs=1)
    """
    Your code here
    """

    BATCH_SIZE = 8

    train_data_path = "datasets/vehicle/train/"
    valid_data_path = "datasets/vehicle/validation/"

    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),transforms.ColorJitter(), 
    transform_train = transforms.Compose([
                                    transforms.Resize((256,256)),
                                    transforms.RandomCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    transform_valid = transforms.Compose([
                                    transforms.Resize((256,256)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])
    
    train_data = VehicleClassificationDataset(train_data_path, transform=transform_train)
    valid_data = VehicleClassificationDataset(valid_data_path, transform=transform_valid)

    print(f"Train on {train_data.__len__()} samples, Validate on {valid_data.__len__()} samples")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = SoftmaxCrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5) #weight_decay=5e-5
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=5e-6, weight_decay=1e-5 )

    confusion = ConfusionMatrix(size=6)

    patience = 5
    v_loss_min = 1_000_000
    counter = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"run model on {device}")
    
    # for parameter in model.resnet_model.parameters():
    #     parameter.requires_grad = False

    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    model.cuda()
    model.train(True)
    EPOCHS = 30
    for epoch in range(EPOCHS):
        print(f"EPOCH [{epoch + 1}/{EPOCHS}] for train... with {len(train_loader)}")        
        loss_sum = 0
        loss_avg = 0

        acc_sum = 0
        acc_avg = 0
        for i, data in enumerate(train_loader):
            X, t = data
            X, t = X.cuda(), t.cuda()

            pred = model(X)            
            optimizer.zero_grad()
            loss = loss_fn(pred, t)
            
            if torch.any(torch.isnan(loss)):
                print("error with nan")
                break

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
            #logits =  F.softmax(pred)
            pred_max = pred.argmax(1) #torch.argmax(pred, dim=1)
            confusion.add(pred_max, t)
            acc = confusion.global_accuracy
            acc_sum += acc
            
            if (i+1) % 10 == 0:
                loss_avg = loss_sum / (i+1)
                acc_avg = acc_sum / (i+1)

                tb_x = epoch * len(train_loader) + i + 1
                train_logger.add_scalar('train/loss', loss_avg, tb_x)
                train_logger.add_scalar('train/acc', acc_avg, tb_x)
                print(f"  [Batch {i+1}/{len(train_loader)}] Loss : {loss_avg:.4f}, Accuracy : {(acc_avg*100):.2f}%")
        
        loss_avg = loss_sum / (i+1)
        acc_avg = acc_sum / (i+1)
        print(f"  [Train] Loss average : {loss_avg:.4f}, Accuracy average : {(acc_avg*100):.2f}%")

        v_loss_sum = 0.0
        v_acc_sum = 0.0
        v_confusion = ConfusionMatrix(size=6)
        
        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valid_loader):
                v_X, v_t = v_data
                v_X, v_t = v_X.cuda(), v_t.cuda()
                v_pred = model(v_X)

                v_loss = loss_fn(v_pred, v_t)
                v_loss_sum += v_loss

                # logits = F.softmax(v_pred)
                v_pred_max = v_pred.argmax(1)
                v_confusion.add(v_pred_max, v_t)
                v_acc_sum += v_confusion.global_accuracy

                v_loss_avg = v_loss_sum / (i+1)
                v_acc_avg = v_acc_sum / (i+1)
                tb_v = epoch * len(valid_loader) + i + 1

                valid_logger.add_scalar('valid/loss', v_loss_avg, tb_v)
                valid_logger.add_scalar('valid/acc', v_acc_avg, tb_v)


        print(f"  [Valid] Loss average : {v_loss_avg:.4f}, Accuracy average : {(v_acc_avg * 100):.2f}% \n")

        train_logger.flush()
        valid_logger.flush()

        if  v_loss_avg < v_loss_min + 5e-4: #v_iou_best < v_iou_avg and
            print(f"save checkpoint....\n") # iou best Update! v_iou_best : {v_iou_best} vs v_iou_avg : {v_iou_avg} 
            v_loss_min = v_loss_avg
            counter = 0
            torch.save(model.state_dict(), 'pretrained/cnn.th')
        elif v_loss_avg >= v_loss_min: #abs(v_iou_avg-v_iou_best) < 5e-4 or v_iou_avg < v_iou_best or
            counter += 1
            print(f"[counter/patience] : [{counter}/{patience}]... \n")

        if counter > patience:
            print(f"Number of counter exceeds patience... early stopping")
            break   

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
