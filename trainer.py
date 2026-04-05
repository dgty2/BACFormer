import torch
from tqdm import tqdm
import numpy as np
# 修正导入路径，metrics在core文件夹里
from core.metrics import calculate_dice


class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs)
        self.device = torch.device("cpu")  # 默认CPU，会被train.py覆盖

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc="训练中")
        for images, labels in pbar:
            # 把数据移到和模型一样的设备
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