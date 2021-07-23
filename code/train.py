import os
import sys
sys.path.append('./code')

import math
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from network.vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate

from dataLoaders.pic_dataset import *
from dataLoaders.video_dataset import *


def img_dataLoader(args, data_transform):
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


def video_dataLoader(args, data_transform):

    train_data = VideoDataset(dataset='DMD-lite-70', split='train', clip_len=args.frames_len, preprocess=False)
    val_data = VideoDataset(dataset='DMD-lite-70', split='val', clip_len=args.frames_len, preprocess=False)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=nw)

    return train_loader, val_loader


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    if args.data_type == 'img':
        train_loader, val_loader = img_dataLoader(args, data_transform)
    
    if args.data_type == 'video':
        train_loader, val_loader = video_dataLoader(args, data_transform)

    model = create_model(num_classes=10, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
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

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--val-size', type=int, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-type', type=str, default='video')
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 视频相关
    parser.add_argument('--frames-len', type=int, default=70)
    

    opt = parser.parse_args()

    main(opt)
