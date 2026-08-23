"""
配置文件：策略超参数与路径配置
"""
from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class DataConfig:
    """数据配置"""
    # 时间范围
    train_start: str = "2010-01-01"
    train_end: str = "2018-12-31"
    valid_start: str = "2019-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2024-12-31"
    
    # 数据源
    data_source: str = "tushare"  # tushare, akshare, joinquant
    
    # 因子配置
    regime_factors: List[str] = None  # 市场状态因子列表
    stock_factors: List[str] = None   # 个股选择因子列表
    
    # 时间窗口
    lookback_window: int = 20  # 回看周数
    rebalance_freq: str = "W-MON"  # 每周一调仓
    
    def __post_init__(self):
        if self.regime_factors is None:
            self.regime_factors = [
                # 宏观因子
                "m2_yoy", "social_financing_yoy", "pmi", "cpi", "ppi",
                "bond_yield_10y", "credit_spread",
                # 市场情绪因子
                "market_volume_chg", "market_turnover", "limit_up_ratio",
                "margin_balance", "northbound_flow", "market_breadth",
                # 技术因子
                "index_macd", "index_rsi", "index_ma_ratio"
            ]
        
        if self.stock_factors is None:
            self.stock_factors = [
                # 估值因子
                "pe_ttm", "pb", "ps_ttm", "pcf_ttm",
                # 盈利因子
                "roe", "roa", "gross_margin", "net_margin",
                # 成长因子
                "revenue_growth", "profit_growth", "roe_growth",
                # 质量因子
                "debt_to_asset", "current_ratio", "receivable_turnover",
                # 动量因子
                "return_1m", "return_3m", "return_6m", "return_12m",
                # 波动因子
                "volatility_20d", "beta_60d",
                # 流动性因子
                "turnover_20d", "amihud_illiquidity",
                # 反转因子
                "return_1w"
            ]


@dataclass
class ModelConfig:
    """模型配置"""
    # Transformer参数
    d_model: int = 128  # 嵌入维度
    n_heads: int = 8    # 注意力头数
    n_layers: int = 4   # Transformer层数
    d_ff: int = 512     # 前馈网络维度
    dropout: float = 0.1
    
    # 因子数量参数（从DataConfig获取）
    n_market_factors: int = None
    n_stock_factors: int = None
    
    # 任务头参数
    regime_classes: int = 3  # 市场状态：牛市/震荡/熊市
    ranking_output: int = 1  # 预测收益率（回归任务）
    
    # 多任务学习权重
    regime_loss_weight: float = 0.4
    ranking_loss_weight: float = 0.6
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """初始化后处理：设置因子数量"""
        if self.n_market_factors is None:
            self.n_market_factors = len(data_config.regime_factors)
        if self.n_stock_factors is None:
            self.n_stock_factors = len(data_config.stock_factors)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 滚动训练
    rolling_retrain: bool = True
    retrain_freq: str = "yearly"  # yearly, quarterly
    
    # 保存路径
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class BacktestConfig:
    """回测配置"""
    # 组合构建
    top_n_stocks: int = 30  # 选取前30只股票
    weight_method: str = "equal"  # equal, risk_parity, optimized
    
    # 仓位控制（基于市场状态）
    position_bull: Tuple[float, float] = (0.8, 1.0)    # 牛市仓位范围
    position_neutral: Tuple[float, float] = (0.4, 0.6)  # 震荡仓位范围
    position_bear: Tuple[float, float] = (0.0, 0.2)    # 熊市仓位范围
    
    # 交易成本
    commission_rate: float = 0.00025  # 万分之2.5
    slippage_rate: float = 0.001      # 0.1%冲击成本
    
    # 风险控制
    max_single_weight: float = 0.1    # 单只股票最大权重10%
    max_sector_weight: float = 0.3    # 单个行业最大权重30%
    
    # 对冲配置（可选）
    use_hedge: bool = False
    hedge_ratio: float = 1.0
    
    # 评估指标
    benchmark: str = "000300.SH"  # 沪深300指数
    risk_free_rate: float = 0.03  # 无风险利率3%


# 全局配置实例
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
backtest_config = BacktestConfig()
