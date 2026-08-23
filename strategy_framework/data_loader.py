"""
数据加载模块：获取、处理、构建数据集
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入tushare，如果失败则设为None
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    ts = None
    TUSHARE_AVAILABLE = False
    print("警告: tushare未安装，将使用模拟数据。安装命令: pip install tushare")

class FactorDataLoader:
    """因子数据加载器"""
    
    def __init__(self, config):
        self.config = config
        # 初始化数据源（需要设置token）
        # ts.set_token('your_tushare_token')
        self.pro = None  # ts.pro_api()
        
    def load_index_components(self, trade_date: str) -> List[str]:
        """
        获取指定日期的沪深300成分股
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            
        Returns:
            股票代码列表
        """
        try:
            if self.pro is None:
                print("警告: Tushare未初始化，返回模拟数据")
                # 返回模拟的300只股票代码
                return [f"{str(i).zfill(6)}.SH" if i % 2 == 0 else f"{str(i).zfill(6)}.SZ" 
                       for i in range(1, 301)]
            
            df = self.pro.index_weight(
                index_code='000300.SH',
                trade_date=trade_date
            )
            if df is not None and len(df) > 0:
                return df['con_code'].tolist()
            else:
                print(f"警告: 未获取到{trade_date}的成分股数据")
                return []
        except Exception as e:
            print(f"加载成分股失败: {e}")
            return []
    
    def load_market_factors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载市场层面因子（用于市场状态识别）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with columns: [trade_date, factor1, factor2, ...]
        """
        # TODO: 实现宏观数据、市场情绪数据获取
        # 示例结构
        factors = {}
        
        # 1. 宏观因子
        # factors['m2_yoy'] = self.load_macro_indicator('M2', start_date, end_date)
        # factors['pmi'] = self.load_macro_indicator('PMI', start_date, end_date)
        
        # 2. 市场情绪因子
        # factors['market_turnover'] = self.calculate_market_turnover(start_date, end_date)
        # factors['northbound_flow'] = self.load_northbound_flow(start_date, end_date)
        
        # 3. 技术因子
        # index_data = self.load_index_daily('000300.SH', start_date, end_date)
        # factors['index_macd'] = self.calculate_macd(index_data)
        # factors['index_rsi'] = self.calculate_rsi(index_data)
        
        # 合并所有因子
        # df = pd.DataFrame(factors)
        # return df
        pass
    
    def load_stock_factors(self, stock_codes: List[str], 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载个股层面因子
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with columns: [trade_date, ts_code, factor1, factor2, ...]
        """
        # TODO: 实现个股因子计算
        all_data = []
        
        for code in stock_codes:
            # 1. 基本面因子（财务数据）
            # financial = self.load_financial_data(code, start_date, end_date)
            # valuation = self.calculate_valuation_factors(financial)
            # profitability = self.calculate_profitability_factors(financial)
            # growth = self.calculate_growth_factors(financial)
            # quality = self.calculate_quality_factors(financial)
            
            # 2. 量价因子
            # daily_data = self.load_daily_data(code, start_date, end_date)
            # momentum = self.calculate_momentum_factors(daily_data)
            # volatility = self.calculate_volatility_factors(daily_data)
            # liquidity = self.calculate_liquidity_factors(daily_data)
            
            # stock_df = pd.concat([valuation, profitability, growth, 
            #                       quality, momentum, volatility, liquidity], axis=1)
            # stock_df['ts_code'] = code
            # all_data.append(stock_df)
            pass
        
        # return pd.concat(all_data, ignore_index=True)
        pass
    
    def calculate_labels(self, stock_data: pd.DataFrame, 
                        forward_window: int = 1) -> pd.DataFrame:
        """
        计算标签：未来收益率、市场状态
        
        Args:
            stock_data: 股票数据
            forward_window: 前瞻窗口（周）
            
        Returns:
            添加标签后的DataFrame
        """
        # 1. 个股未来收益率（用于排序任务）
        stock_data['future_return'] = stock_data.groupby('ts_code')['close'].pct_change(forward_window).shift(-forward_window)
        
        # 2. 市场状态标签（用于择时任务）
        # 基于沪深300指数未来收益率分类
        # index_return = ...
        # conditions = [
        #     index_return > threshold_bull,
        #     index_return < threshold_bear
        # ]
        # choices = [2, 0]  # 2=牛市, 1=震荡, 0=熊市
        # stock_data['market_regime'] = np.select(conditions, choices, default=1)
        
        return stock_data
    
    def preprocess_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        因子预处理：标准化、去极值、中性化
        
        Args:
            df: 原始因子数据
            
        Returns:
            处理后的因子数据
        """
        # 1. 去极值（MAD方法）
        def winsorize_mad(series, n=3):
            median = series.median()
            mad = (series - median).abs().median()
            upper = median + n * mad * 1.4826
            lower = median - n * mad * 1.4826
            return series.clip(lower, upper)
        
        # 2. 标准化（Z-score）
        def standardize(series):
            return (series - series.mean()) / series.std()
        
        # 3. 行业中性化（可选）
        def neutralize_industry(df, factor_col, industry_col):
            # 回归去除行业影响
            # residuals = factor - industry_dummies @ beta
            pass
        
        # 对每个因子应用预处理
        factor_cols = [col for col in df.columns if col not in ['trade_date', 'ts_code']]
        for col in factor_cols:
            df[col] = df.groupby('trade_date')[col].transform(winsorize_mad)
            df[col] = df.groupby('trade_date')[col].transform(standardize)
        
        return df


class MultiFactorDataset(Dataset):
    """多因子数据集（PyTorch Dataset）"""
    
    def __init__(self, market_factors: np.ndarray, stock_factors: np.ndarray,
                 regime_labels: np.ndarray, ranking_labels: np.ndarray):
        """
        Args:
            market_factors: [n_samples, seq_len, n_market_factors]
            stock_factors: [n_samples, n_stocks, n_stock_factors]
            regime_labels: [n_samples] 市场状态标签
            ranking_labels: [n_samples, n_stocks] 个股收益率标签
        """
        self.market_factors = torch.FloatTensor(market_factors)
        self.stock_factors = torch.FloatTensor(stock_factors)
        self.regime_labels = torch.LongTensor(regime_labels)
        self.ranking_labels = torch.FloatTensor(ranking_labels)
    
    def __len__(self):
        return len(self.regime_labels)
    
    def __getitem__(self, idx):
        return {
            'market_factors': self.market_factors[idx],
            'stock_factors': self.stock_factors[idx],
            'regime_label': self.regime_labels[idx],
            'ranking_label': self.ranking_labels[idx]
        }


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    print("警告: create_dataloaders使用模拟数据")
    print("请实现真实的数据加载逻辑后再进行训练")
    
    # 创建模拟数据用于测试代码结构
    n_samples = 100
    seq_len = config.lookback_window
    n_stocks = 300
    n_market_factors = len(config.regime_factors)
    n_stock_factors = len(config.stock_factors)
    
    # 生成随机模拟数据
    market_factors = np.random.randn(n_samples, seq_len, n_market_factors)
    stock_factors = np.random.randn(n_samples, n_stocks, n_stock_factors)
    regime_labels = np.random.randint(0, 3, n_samples)  # 0,1,2三个类别
    ranking_labels = np.random.randn(n_samples, n_stocks) * 0.1  # 模拟收益率
    
    # 创建数据集
    dataset = MultiFactorDataset(
        market_factors, stock_factors,
        regime_labels, ranking_labels
    )
    
    # 划分训练/验证/测试集
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, n_samples))
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"数据加载完成: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}, 测试集{len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
