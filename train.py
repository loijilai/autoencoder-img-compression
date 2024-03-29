import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

# Local imports
from model.components.autoencoder import Autoencoder
from utils.dataset import imgDataset


# Parameters for training
def parse_args():
    dataset_path = "./dataset"
    checkpoint_path = None
    save_at = "./out"
    save_frequency = 10
    file_format = 'bmp'

    num_of_epochs = 200
    batch_size = 25
    validation_split = 0.1
    lr_rate = 0.001

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=dataset_path, help='Root directory of Images')
    parser.add_argument('--checkpoint_path', default=checkpoint_path, help='Use to resume training from last checkpoint')
    parser.add_argument('--save_at', default=save_at, help='Directory where training state will be saved')
    parser.add_argument('--save_frequency', default=save_frequency, help='Save checkpoint every n epochs', type=int)
    parser.add_argument('--file_format', default=file_format, help='File format of images to train on')

    parser.add_argument('--num_of_epochs', default=num_of_epochs,help='Epoch to stop training at',type=int)
    parser.add_argument('--batch_size', default=batch_size, help='Batch size for training', type=int)
    parser.add_argument('--validation_split', default=validation_split, help='Validation split for training', type=float)
    parser.add_argument('--lr_rate', default=lr_rate, help='Learning rate for training', type=float)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Handle directory
    if not os.path.exists(args.save_at):
        os.makedirs(args.save_at)

    model = Autoencoder().float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    # TODO
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1)
    model = model.cuda()

    history = {
        'train_losses':[],
        'val_losses' :[],
        'epoch_data' : [],
    }

    # If train from checkpoint
    if(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        history = checkpoint['history']
        start_epoch = checkpoint['history']['epoch_data'][-1] + 1
        print(f"Continue training at epoch: {start_epoch}")
    else:
        start_epoch = 1

    # Dataset & Dataloader
    raw_dataset = imgDataset(root_dir=args.dataset_path, file_format=args.file_format)
    dataset_size = len(raw_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))
    np.random.shuffle(indices) # Shuffle dataset
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=args.batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=args.batch_size, sampler=validation_sampler)
    print("====== Dataset Information ======")
    print(f"Train dataset size: {len(train_indices)}")
    print(f"Validation dataset size: {len(val_indices)}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    # Start training
    print("====== Start training...======")
    epoch_to_train = args.num_of_epochs - start_epoch + 1
    total_batch = args.num_of_epochs * len(train_loader)
    start_batch = (start_epoch-1) * len(train_loader)
    pbar = tqdm(total=total_batch, initial=start_batch, desc = "Training progress")
    for epoch in range(start_epoch, start_epoch+epoch_to_train):
        model.train()
        train_loss = 0
        # train
        for batch_idx, data in enumerate(train_loader):
            target = data
            data = data.cuda()
            target = target.cuda()

            # forward
            output = model(data)

            # backward + optimize
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            train_loss += loss.item()
            pbar.update(1)
        print(f'Epoch: {epoch}\tTrain loss: {train_loss/len(train_indices)}')

        # evaluate
        model.eval() 
        eval_loss = 0
        for batch_idx, data in enumerate(validation_loader):
            target = data
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = criterion(output, target)
            eval_loss += loss.item()
        print(f'Epoch: {epoch}\tEval loss: {eval_loss/len(val_indices)}')

        # save statistics and checkpoint
        history['train_losses'].append(train_loss/len(train_loader))
        history['val_losses'].append(eval_loss/len(validation_loader))
        history['epoch_data'].append(epoch)
        if epoch % args.save_frequency == 0:
            print(f"====== Saving checkpoint to {args.save_at}... ======")
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
            }
            torch.save(checkpoint, os.path.join(args.save_at, f"checkpoint_{epoch}.pth"))

    pbar.close()
    print("Finish training...")

    # save final checkpoint
    print(f"====== Saving checkpoint to {args.save_at}... ======")
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
    }

    torch.save(checkpoint, os.path.join(args.save_at, "final_checkpoint.pth"))
    print("Finish!")

if __name__ == '__main__':
    main()