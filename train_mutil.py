import os
import time
import datetime
from typing import Union, List

import torch
import torch.nn as nn
from dataset import MultiModalDataset
from torch.utils import data

# from data.data_pre import MultiModalDataset
from model import VIT, Transformer, MultimodalModel
# from train_utils import  train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from train_utils import (train_one_epoch, evaluate, init_distributed_mode, save_on_master, mkdir,
                         create_lr_scheduler, get_params_groups)

def main(args):
    init_distributed_mode(args)
    print(args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = MultiModalDataset(args.audio_dir, args.image_dir, args.train_label_file, phase='train') 
    val_dataset = MultiModalDataset(args.audio_dir, args.image_dir, args.val_label_file, phase='val')
    print("Creating data loaders")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    
    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        test_sampler = data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = data.RandomSampler(train_dataset)
        test_sampler = data.SequentialSampler(val_dataset)

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    

    val_data_loader = data.DataLoader(
        val_dataset, batch_size=1,  # batch_size must be 1
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=val_dataset.collate_fn)
    print(f"Number of training steps per epoch: {len(train_data_loader)}")
    
    model = MultimodalModel(num_classes=3).to(device)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    

    
    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)
    

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    
    
    print("Start training")
    current_f1 = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            f1_metric = evaluate(model, val_data_loader, device=device)
            f1_info = f1_metric.compute()
            print(f"[epoch: {epoch}] val_maxF1: {f1_info:.3f}")
            
            # 只在主进程上进行写操作
            if args.rank in [-1, 0]:
                # write into txt
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"maxF1: {f1_info:.3f} \n"
                    f.write(write_info)
            
                if current_f1 <= f1_info:
                    if args.output_dir:
                        # 只在主节点上执行保存权重操作
                        save_on_master(save_file,
                                       os.path.join(args.output_dir, 'model_best.pth'))


        if args.output_dir:
            if args.rank in [-1, 0]:
                # only save latest 10 epoch weights
                if os.path.exists(os.path.join(args.output_dir, f'model_{epoch - 10}.pth')):
                    os.remove(os.path.join(args.output_dir, f'model_{epoch - 10}.pth'))

            # 只在主节点上执行保存权重操作
            save_on_master(save_file,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Multimodal Training")

    # 数据路径参数
    parser.add_argument("--audio-dir", default="./output_audio", help="Audio data directory")
    parser.add_argument("--image-dir", default="./output_images", help="Image data directory")
    parser.add_argument("--train-label-file", default="./label.csv", help="Training labels file")
    parser.add_argument("--val-label-file", default="./val_label.csv", help="Validation labels file")
    
    # 训练设置参数
    parser.add_argument('--device', default='cuda', help='training device')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--sync-bn', action='store_true', help='whether using SyncBatchNorm')
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式训练参数
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    # 混合精度训练参数
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)
    
    # 运行主函数
    main(args)