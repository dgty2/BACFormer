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
    # 创建参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description="BACFormer 左心房分割训练")
    
    # 添加数据集名称参数
    parser.add_argument("--dataset", type=str, default="LeftAtrium")
    # 添加数据根目录路径参数
    parser.add_argument("--root_path", type=str, default="./")
    # 添加列表文件目录参数
    parser.add_argument("--list_dir", type=str, default="./lists/lists_LeftAtrium")
    # 添加最大训练轮数参数
    parser.add_argument("--max_epochs", type=int, default=50)
    # 添加模型输出目录参数
    parser.add_argument("--output_dir", type=str, default="weights")
    # 添加输入图像尺寸参数
    parser.add_argument("--img_size", type=int, default=224)
    # 添加批次大小参数
    parser.add_argument("--batch_size", type=int, default=4)
    # 添加基础学习率参数
    parser.add_argument("--base_lr", type=float, default=0.001)
    # 添加验证间隔参数（每隔多少epoch验证一次）
    parser.add_argument("--val_epoch", type=int, default=10)
    # 添加分割类别数参数（背景+左心房）
    parser.add_argument("--num_classes", type=int, default=2)
    # 添加断点续训权重路径参数
    parser.add_argument("--resume", type=str, default=None)
    
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 根据CUDA可用性自动选择设备（GPU优先，否则CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印训练开始信息和使用的设备
    print(f"开始训练，设备：{device}")

    # 检查输出目录是否存在
    if not os.path.exists(args.output_dir):
        # 如果不存在，创建输出目录用于保存模型权重
        os.makedirs(args.output_dir)

    # 创建训练数据集实例
    train_dataset = Dataset_LeftAtrium(root_path=args.root_path, list_dir=args.list_dir, split="train",
                                       img_size=args.img_size)
    # 创建验证数据集实例
    val_dataset = Dataset_LeftAtrium(root_path=args.root_path, list_dir=args.list_dir, split="test",
                                     img_size=args.img_size)
    
    # 创建训练数据加载器，启用随机打乱
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # 创建验证数据加载器，不打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 创建BACFormer模型实例
    model = BACFormer(num_classes=args.num_classes)
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)

    # 如果指定了断点续训路径
    if args.resume:
        # 打印加载权重信息
        print(f"加载断点权重：{args.resume}")
        # 加载权重文件到指定设备
        checkpoint = torch.load(args.resume, map_location=device)
        # 将权重加载到模型中
        model.load_state_dict(checkpoint)

    # 创建训练器实例，传入模型、数据加载器和参数
    trainer = Trainer(model, train_loader, val_loader, args)
    # 将设备信息传递给训练器
    trainer.device = device
    # 开始执行训练流程
    trainer.train()