"""
模型定义：多任务Transformer架构
"""
import torch
import torch.nn as nn
import math
from config import data_config

class PositionalEncoding(nn.Module):
    """位置编码（用于时序信息）"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FactorEmbedding(nn.Module):
    """因子嵌入层"""
    
    def __init__(self, n_factors: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(n_factors, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, n_factors]
        Returns:
            [batch, seq_len, d_model]
        """
        return self.norm(self.linear(x))


class MultiTaskTransformer(nn.Module):
    """
    多任务学习Transformer模型
    同时完成：1) 市场状态识别  2) 个股收益预测
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 市场因子嵌入（用于择时）
        self.market_embedding = FactorEmbedding(
            n_factors=len(data_config.regime_factors),
            d_model=config.d_model
        )
        
        # 个股因子嵌入（用于选股）
        self.stock_embedding = FactorEmbedding(
            n_factors=len(data_config.stock_factors),
            d_model=config.d_model
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Transformer编码器（共享）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # 任务1：市场状态分类头
        self.regime_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.regime_classes)
        )
        
        # 任务2：个股排序头
        self.ranking_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.ranking_output)
        )
        
        # 创新点：因子动态加权模块（可选）
        self.use_dynamic_weighting = False
        if self.use_dynamic_weighting:
            self.factor_weight_generator = nn.Sequential(
                nn.Linear(config.d_model, len(data_config.stock_factors)),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, market_factors, stock_factors):
        """
        Args:
            market_factors: [batch, seq_len, n_market_factors]
            stock_factors: [batch, n_stocks, n_stock_factors]
            
        Returns:
            regime_logits: [batch, regime_classes]
            ranking_scores: [batch, n_stocks]
        """
        batch_size = market_factors.size(0)
        
        # === 任务1：市场状态识别 ===
        # 嵌入市场因子
        market_embed = self.market_embedding(market_factors)  # [batch, seq_len, d_model]
        
        # 添加位置编码
        market_embed = market_embed.transpose(0, 1)  # [seq_len, batch, d_model]
        market_embed = self.pos_encoder(market_embed)
        market_embed = market_embed.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Transformer编码
        market_encoded = self.transformer_encoder(market_embed)  # [batch, seq_len, d_model]
        
        # 取最后时间步用于分类
        market_state = market_encoded[:, -1, :]  # [batch, d_model]
        regime_logits = self.regime_head(market_state)  # [batch, regime_classes]
        
        # === 任务2：个股收益预测 ===
        # 嵌入个股因子
        stock_embed = self.stock_embedding(stock_factors)  # [batch, n_stocks, d_model]
        
        # 创新点：根据市场状态动态调整因子权重
        if self.use_dynamic_weighting:
            factor_weights = self.factor_weight_generator(market_state)  # [batch, n_factors]
            # 应用权重到原始因子
            stock_factors_weighted = stock_factors * factor_weights.unsqueeze(1)
            stock_embed = self.stock_embedding(stock_factors_weighted)
        
        # Transformer编码（可以复用或使用独立编码器）
        stock_encoded = self.transformer_encoder(stock_embed)  # [batch, n_stocks, d_model]
        
        # 预测每只股票的收益
        ranking_scores = self.ranking_head(stock_encoded).squeeze(-1)  # [batch, n_stocks]
        
        return regime_logits, ranking_scores
    
    def get_attention_weights(self):
        """获取注意力权重（用于可视化）"""
        # TODO: 实现注意力权重提取
        pass


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, regime_weight: float = 0.4, ranking_weight: float = 0.6):
        super().__init__()
        self.regime_weight = regime_weight
        self.ranking_weight = ranking_weight
        
        # 分类损失（市场状态）
        self.regime_loss_fn = nn.CrossEntropyLoss()
        
        # 排序损失（个股收益）- 使用Ranking Loss
        self.ranking_loss_fn = self.pairwise_ranking_loss
    
    def pairwise_ranking_loss(self, pred_scores, true_returns, margin=0.01):
        """
        成对排序损失：确保高收益股票的预测分数 > 低收益股票
        
        Args:
            pred_scores: [batch, n_stocks] 预测分数
            true_returns: [batch, n_stocks] 真实收益率
            margin: 边界值
        """
        batch_size, n_stocks = pred_scores.shape
        
        # 构建成对比较
        # pred_diff[i,j,k] = pred_scores[i,j] - pred_scores[i,k]
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # [batch, n_stocks, n_stocks]
        
        # true_diff[i,j,k] = true_returns[i,j] - true_returns[i,k]
        true_diff = true_returns.unsqueeze(2) - true_returns.unsqueeze(1)  # [batch, n_stocks, n_stocks]
        
        # 只考虑收益差异显著的股票对
        mask = (true_diff.abs() > margin).float()
        
        # Hinge loss: max(0, -sign(true_diff) * pred_diff + margin)
        loss = torch.relu(-torch.sign(true_diff) * pred_diff + margin)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def forward(self, regime_logits, ranking_scores, regime_labels, ranking_labels):
        """
        计算总损失
        
        Args:
            regime_logits: [batch, regime_classes]
            ranking_scores: [batch, n_stocks]
            regime_labels: [batch]
            ranking_labels: [batch, n_stocks]
        """
        # 择时损失
        regime_loss = self.regime_loss_fn(regime_logits, regime_labels)
        
        # 选股损失
        ranking_loss = self.ranking_loss_fn(ranking_scores, ranking_labels)
        
        # 加权总损失
        total_loss = (self.regime_weight * regime_loss + 
                     self.ranking_weight * ranking_loss)
        
        return total_loss, regime_loss, ranking_loss
