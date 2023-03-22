from models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    """
    Your code here
    """

    BATCH_SIZE = 32

    train_data_path = "datasets/vehicle/train/"
    valid_data_path = "datasets/vehicle/validation/"
    # dataset = VehicleClassificationDataset(dataset_path)

    # train_size = int(0.8 * len(dataset))
    # valid_size = int(0.2 * len(dataset)) + 1

    #train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_data = VehicleClassificationDataset(train_data_path);
    valid_data = VehicleClassificationDataset(valid_data_path)

    print(f"Train on {train_data.__len__()} samples , Validate on {valid_data.__len__()} samples")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 5
    v_loss_min = 1_000_000.
    v_loss = 0

    for epoch in range(EPOCHS):

        model.train(True)

        epoch_loss = 0.0
        #loss_sum = 0

        loss_list = []
        
        print(f"EPOCH {epoch + 1} for train... with {len(train_loader)}")

        for i, data in enumerate(train_loader):
            X, t = data
            pred = model(X)
            loss = model.loss
            
            optimizer.zero_grad()

            loss = loss_fn(pred, t)
            
            if loss.isnan():
                print("error with nan")
                break

            #print(f"Loss: {loss.item()}")

            loss.backward()

            #print("backward!")

            optimizer.step()

            #print("optimzier!")

            epoch_loss += loss.item()

            #print("sum")
            if (i+1) % 10 == 0 or (i+1) == BATCH_SIZE:
                # print(f"size of X: {X.shape}")
                loss_avg = epoch_loss / (i+1)
                loss_list.append(loss_avg)
                #print(f"shape of loss_list {len(loss_list)}")
                print(f"  Batch {i+1} / {len(train_loader)} Loss : {loss_avg}")
                tb_x = epoch * len(train_loader) + i + 1
                train_logger.add_scalar('train/loss', loss_avg, tb_x)

        
        loss_avg = sum(loss_list) / len(loss_list)
        print(f"  [Train] Loss : {loss_avg}")
        loss_list.clear()
        loss_avg = 0

        print(f"EPOCH {epoch+1} for validation...")
        for i, valid_data in enumerate(valid_loader):
            v_X, v_t = valid_data

            v_pred = model(v_X)
            v_loss = loss_fn(v_pred, v_t)
            print(f"  {i+1} validation....")
            if v_loss_min > v_loss:
              v_loss_min = v_loss

        print(f"  [Valid] Loss : {v_loss_min} \n")

        valid_logger.add_scalar('valid/loss', v_loss_min, epoch+1)

        train_logger.flush()
        valid_logger.flush()

        # if v_loss_avg < v_loss_min:

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
