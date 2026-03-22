import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from core.metrics import DiceBoundaryLoss, calculate_dice


class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = DiceBoundaryLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs)

        self.best_dice = 0.0
        os.makedirs(args.output_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, masks in tqdm(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device).long()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def val_epoch(self):
        self.model.eval()
        total_dice = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device).long()
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                for pred, mask in zip(preds, masks):
                    pred_np = pred.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    dice = calculate_dice(pred_np, mask_np)
                    total_dice += dice

        return total_dice / len(self.val_loader.dataset)

    def train(self):
        print(f"开始训练，设备：{self.device}")
        for epoch in range(1, self.args.max_epochs + 1):
            train_loss = self.train_epoch()
            val_dice = self.val_epoch()
            self.scheduler.step()

            print(f"Epoch {epoch}/{self.args.max_epochs} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "best_model.pth"))
                print(f"✅ 最优权重已保存，Dice: {self.best_dice:.4f}")

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, f"epoch_{epoch}.pth"))

        print(f"训练完成！最优Dice: {self.best_dice:.4f}")