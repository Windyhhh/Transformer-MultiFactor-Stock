"""
训练模块：模型训练、验证、早停
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Tuple

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_epochs
            )
        elif config.lr_scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5
            )
        
        # 损失函数
        from model import MultiTaskLoss
        from config import model_config
        self.criterion = MultiTaskLoss(
            regime_weight=model_config.regime_loss_weight,
            ranking_weight=model_config.ranking_loss_weight
        )
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 日志
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        regime_loss_sum = 0
        ranking_loss_sum = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # 数据移到设备
            market_factors = batch['market_factors'].to(self.device)
            stock_factors = batch['stock_factors'].to(self.device)
            regime_labels = batch['regime_label'].to(self.device)
            ranking_labels = batch['ranking_label'].to(self.device)
            
            # 前向传播
            regime_logits, ranking_scores = self.model(market_factors, stock_factors)
            
            # 计算损失
            loss, regime_loss, ranking_loss = self.criterion(
                regime_logits, ranking_scores,
                regime_labels, ranking_labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            regime_loss_sum += regime_loss.item()
            ranking_loss_sum += ranking_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'regime': f'{regime_loss.item():.4f}',
                'ranking': f'{ranking_loss.item():.4f}'
            })
        
        n_batches = len(train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'regime_loss': regime_loss_sum / n_batches,
            'ranking_loss': ranking_loss_sum / n_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        regime_loss_sum = 0
        ranking_loss_sum = 0
        
        # 用于计算指标
        all_regime_preds = []
        all_regime_labels = []
        all_ranking_scores = []
        all_ranking_labels = []
        
        for batch in tqdm(val_loader, desc='Validating'):
            market_factors = batch['market_factors'].to(self.device)
            stock_factors = batch['stock_factors'].to(self.device)
            regime_labels = batch['regime_label'].to(self.device)
            ranking_labels = batch['ranking_label'].to(self.device)
            
            # 前向传播
            regime_logits, ranking_scores = self.model(market_factors, stock_factors)
            
            # 计算损失
            loss, regime_loss, ranking_loss = self.criterion(
                regime_logits, ranking_scores,
                regime_labels, ranking_labels
            )
            
            total_loss += loss.item()
            regime_loss_sum += regime_loss.item()
            ranking_loss_sum += ranking_loss.item()
            
            # 收集预测结果
            all_regime_preds.append(regime_logits.argmax(dim=1).cpu().numpy())
            all_regime_labels.append(regime_labels.cpu().numpy())
            all_ranking_scores.append(ranking_scores.cpu().numpy())
            all_ranking_labels.append(ranking_labels.cpu().numpy())
        
        # 计算评估指标
        regime_preds = np.concatenate(all_regime_preds)
        regime_labels = np.concatenate(all_regime_labels)
        regime_acc = (regime_preds == regime_labels).mean()
        
        ranking_scores = np.concatenate(all_ranking_scores, axis=0)
        ranking_labels = np.concatenate(all_ranking_labels, axis=0)
        ic = self.calculate_ic(ranking_scores, ranking_labels)
        
        n_batches = len(val_loader)
        return {
            'total_loss': total_loss / n_batches,
            'regime_loss': regime_loss_sum / n_batches,
            'ranking_loss': ranking_loss_sum / n_batches,
            'regime_accuracy': regime_acc,
            'ranking_ic': ic
        }
    
    def calculate_ic(self, pred_scores: np.ndarray, true_returns: np.ndarray) -> float:
        """
        计算信息系数（IC）：预测分数与真实收益的相关性
        
        Args:
            pred_scores: [n_samples, n_stocks]
            true_returns: [n_samples, n_stocks]
        """
        from scipy.stats import spearmanr
        
        ics = []
        for i in range(len(pred_scores)):
            # 过滤NaN值
            mask = ~(np.isnan(pred_scores[i]) | np.isnan(true_returns[i]))
            if mask.sum() > 10:  # 至少10只股票
                ic, _ = spearmanr(pred_scores[i][mask], true_returns[i][mask])
                ics.append(ic)
        
        return np.mean(ics) if ics else 0.0
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"Training on {self.device}")
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['total_loss'])
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['total_loss'])
            
            # 打印指标
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Regime Acc: {val_metrics['regime_accuracy']:.4f}")
            print(f"Ranking IC: {val_metrics['ranking_ic']:.4f}")
            
            # 学习率调度
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['total_loss'])
            else:
                self.scheduler.step()
            
            # 早停检查
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """加载模型检查点"""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {path}")
