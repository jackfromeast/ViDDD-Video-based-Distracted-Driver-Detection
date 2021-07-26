import os
import sys
sys.path.append('./code')

import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import train_one_epoch, evaluate

from dataLoaders.pic_dataset import *
from dataLoaders.video_dataset import *

from network.video_vit import create_vvit

import config


args = config.parse_opt()


def img_dataLoader(data_transform):
    data_df = process_data()
    train_df, val_df = train_test_split(data_df, args.val_size, shuffle=True, random_state=1)

    # 实例化训练数据集
    train_dataset = MyDataSet(train_df,
                            transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(val_df,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)
    
    return train_loader, val_loader


def video_dataLoader(data_transform):

    train_data = VideoDataset(dataset='DMD-lite-70', split='train', clip_len=args.frames_len, preprocess=False)
    val_data = VideoDataset(dataset='DMD-lite-70', split='val', clip_len=args.frames_len, preprocess=False)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw,collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=nw,collate_fn=val_data.collate_fn)

    return train_loader, val_loader


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./code/weights") is False:
        os.makedirs("./code/weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

   
    if args.data_type == 'video':
        train_loader, val_loader = video_dataLoader(data_transform)

    model = create_vvit(device).to(device)

    # hyper-parmeters
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)


        print("Training: epoch {}, loss {}, accuracy {}".format(epoch, train_loss, train_acc))
        print("Validing: epoch {}, loss {}, accuracy {}".format(epoch, val_loss, val_acc))
        torch.save(model.state_dict(), "./code/weights/model-{}.pth".format(epoch))



if __name__ == '__main__':

    main()
