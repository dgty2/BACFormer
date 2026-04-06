import torch
from tqdm import tqdm
import numpy as np
from core.metrics import calculate_dice


class Trainer:
    """
    训练器类：负责模型的训练、验证和保存
    
    封装了训练循环、验证循环和模型保存逻辑
    使用余弦退火学习率调度器和Adam优化器
    """
    
    def __init__(self, model, train_loader, val_loader, args):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 待训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            args (argparse.Namespace): 训练参数配置
        """
        # 保存模型实例
        self.model = model
        # 保存训练数据加载器
        self.train_loader = train_loader
        # 保存验证数据加载器
        self.val_loader = val_loader
        # 保存命令行参数配置
        self.args = args
        # 创建Adam优化器，设置学习率为args.base_lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
        # 创建余弦退火学习率调度器，T_max为最大epoch数
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs)
        # 初始化设备为CPU（会被train.py覆盖为GPU）
        self.device = torch.device("cpu")

    def train_epoch(self):
        """
        执行一个epoch的训练
        
        遍历训练数据集，执行前向传播、损失计算、反向传播和参数更新
        
        Returns:
            float: 平均训练损失
        """
        # 将模型设置为训练模式（启用dropout和batchnorm的训练行为）
        self.model.train()
        # 初始化累计训练损失
        train_loss = 0
        # 创建进度条，显示"训练中"
        pbar = tqdm(self.train_loader, desc="训练中")
        
        # 遍历训练数据加载器中的每个批次
        for images, labels in pbar:
            # 将图像数据移动到指定设备
            images = images.to(self.device)
            # 将标签数据移动到指定设备
            labels = labels.to(self.device)

            # 清空之前的梯度
            self.optimizer.zero_grad()
            # 执行模型前向传播，得到预测输出
            outputs = self.model(images)
            # 计算交叉熵损失
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            # 执行反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            self.optimizer.step()

            # 累加当前批次的损失值
            train_loss += loss.item()
            # 在进度条上显示当前损失（保留4位小数）
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 返回平均训练损失
        return train_loss / len(self.train_loader)

    def val_epoch(self):
        """
        执行验证集评估
        
        在验证集上计算Dice系数，用于监控模型性能
        
        Returns:
            float: 平均Dice系数
        """
        # 将模型设置为评估模式（关闭dropout和batchnorm的更新）
        self.model.eval()
        # 初始化累计Dice系数
        val_dice = 0
        
        # 禁用梯度计算，节省内存并加速
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for images, labels in self.val_loader:
                # 将图像数据移动到指定设备
                images = images.to(self.device)
                # 将标签数据移动到指定设备
                labels = labels.to(self.device)

                # 执行模型前向传播
                outputs = self.model(images)
                # 取概率最大的类别作为预测结果
                pred = torch.argmax(outputs, dim=1)

                # 计算预测和真实标签之间的Dice系数
                dice = calculate_dice(pred.cpu().numpy(), labels.cpu().numpy())
                # 累加Dice系数
                val_dice += dice

        # 返回平均Dice系数
        return val_dice / len(self.val_loader)

    def train(self):
        """
        执行完整的训练流程
        
        遍历所有epoch，每个epoch后执行训练和验证
        定期保存最优模型（基于Dice系数）
        """
        # 初始化最优Dice系数为0
        best_dice = 0
        
        # 遍历所有训练轮次
        for epoch in range(self.args.max_epochs):
            # 打印当前epoch信息
            print(f"\nEpoch {epoch + 1}/{self.args.max_epochs}")
            
            # 执行一个epoch的训练
            train_loss = self.train_epoch()
            # 更新学习率（余弦退火）
            self.scheduler.step()

            # 检查是否达到验证间隔
            if (epoch + 1) % self.args.val_epoch == 0:
                # 执行验证集评估
                val_dice = self.val_epoch()
                # 打印验证Dice系数
                print(f"验证Dice: {val_dice:.4f}")
                
                # 如果当前Dice优于最优Dice
                if val_dice > best_dice:
                    # 更新最优Dice
                    best_dice = val_dice
                    # 保存最优模型权重到文件
                    torch.save(self.model.state_dict(), f"{self.args.output_dir}/best_model.pth")
                    # 打印保存信息
                    print(f"保存最优权重，Dice: {best_dice:.4f}")

        # 训练完成后打印最优Dice
        print(f"训练完成，最优Dice: {best_dice:.4f}")