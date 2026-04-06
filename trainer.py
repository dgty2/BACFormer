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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs)
        self.device = torch.device("cpu")

    def train_epoch(self):
        """
        执行一个epoch的训练
        
        遍历训练数据集，执行前向传播、损失计算、反向传播和参数更新
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc="训练中")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return train_loss / len(self.train_loader)

    def val_epoch(self):
        """
        执行验证集评估
        
        在验证集上计算Dice系数，用于监控模型性能
        
        Returns:
            float: 平均Dice系数
        """
        self.model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                pred = torch.argmax(outputs, dim=1)

                dice = calculate_dice(pred.cpu().numpy(), labels.cpu().numpy())
                val_dice += dice

        return val_dice / len(self.val_loader)

    def train(self):
        """
        执行完整的训练流程
        
        遍历所有epoch，每个epoch后执行训练和验证
        定期保存最优模型（基于Dice系数）
        """
        best_dice = 0
        for epoch in range(self.args.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.max_epochs}")
            train_loss = self.train_epoch()
            self.scheduler.step()

            if (epoch + 1) % self.args.val_epoch == 0:
                val_dice = self.val_epoch()
                print(f"验证Dice: {val_dice:.4f}")
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(self.model.state_dict(), f"{self.args.output_dir}/best_model.pth")
                    print(f"保存最优权重，Dice: {best_dice:.4f}")

        print(f"训练完成，最优Dice: {best_dice:.4f}")