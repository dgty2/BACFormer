import argparse
import torch
from datasets.dataset_left_atrium import Dataset_LeftAtrium
from torch.utils.data import DataLoader
from networks.BACFormer import BACFormer
from trainer import Trainer
import os


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="BACFormer 左心房分割训练")
    parser.add_argument("--dataset", type=str, default="LeftAtrium")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_LeftAtrium")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="weights")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--val_epoch", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始训练，设备：{device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_dataset = Dataset_LeftAtrium(root_path=args.root_path, list_dir=args.list_dir, split="train",
                                       img_size=args.img_size)
    val_dataset = Dataset_LeftAtrium(root_path=args.root_path, list_dir=args.list_dir, split="test",
                                     img_size=args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BACFormer(num_classes=args.num_classes)
    model = model.to(device)

    if args.resume:
        print(f"加载断点权重：{args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)

    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.device = device
    trainer.train()