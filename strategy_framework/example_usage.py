"""
使用示例：展示如何使用策略框架
"""
import pandas as pd
import numpy as np
import torch
from config import data_config, model_config, training_config, backtest_config
from factor_library import FactorCalculator

# ============================================================
# 示例1：因子计算
# ============================================================

def example_factor_calculation():
    """示例：计算股票因子"""
    print("=" * 50)
    print("示例1：因子计算")
    print("=" * 50)
    
    # 模拟价格数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    volume = pd.Series(np.random.randint(1000000, 5000000, 100), index=dates)
    
    # 初始化因子计算器
    calc = FactorCalculator()
    
    # 计算动量因子
    momentum_20d = calc.momentum(prices, window=20)
    print(f"\n20日动量因子（最新5个值）：\n{momentum_20d.tail()}")
    
    # 计算RSI
    rsi = calc.rsi(prices, window=14)
    print(f"\nRSI指标（最新5个值）：\n{rsi.tail()}")
    
    # 计算MACD
    macd_line, signal_line, histogram = calc.macd(prices)
    print(f"\nMACD（最新5个值）：\n{macd_line.tail()}")
    
    # 计算波动率
    returns = prices.pct_change()
    volatility = calc.volatility(returns, window=20)
    print(f"\n20日波动率（最新5个值）：\n{volatility.tail()}")


# ============================================================
# 示例2：数据预处理
# ============================================================

def example_data_preprocessing():
    """示例：因子预处理流程"""
    print("\n" + "=" * 50)
    print("示例2：数据预处理")
    print("=" * 50)
    
    # 模拟因子数据
    n_stocks = 300
    n_factors = 10
    
    data = pd.DataFrame({
        'trade_date': ['2023-01-01'] * n_stocks,
        'ts_code': [f'stock_{i:03d}' for i in range(n_stocks)],
    })
    
    # 添加因子（包含极端值和缺失值）
    for i in range(n_factors):
        factor_values = np.random.randn(n_stocks)
        # 添加极端值
        factor_values[0] = 100  # 极大值
        factor_values[1] = -100  # 极小值
        # 添加缺失值
        factor_values[2] = np.nan
        data[f'factor_{i}'] = factor_values
    
    print(f"\n原始数据统计：\n{data.describe()}")
    
    # 去极值（MAD方法）
    def winsorize_mad(series, n=3):
        median = series.median()
        mad = (series - median).abs().median()
        upper = median + n * mad * 1.4826
        lower = median - n * mad * 1.4826
        return series.clip(lower, upper)
    
    # 标准化
    def standardize(series):
        return (series - series.mean()) / series.std()
    
    # 处理因子
    factor_cols = [col for col in data.columns if col.startswith('factor_')]
    for col in factor_cols:
        # 填充缺失值（用中位数）
        data[col].fillna(data[col].median(), inplace=True)
        # 去极值
        data[col] = winsorize_mad(data[col])
        # 标准化
        data[col] = standardize(data[col])
    
    print(f"\n处理后数据统计：\n{data.describe()}")


# ============================================================
# 示例3：模型推理
# ============================================================

def example_model_inference():
    """示例：使用训练好的模型进行预测"""
    print("\n" + "=" * 50)
    print("示例3：模型推理")
    print("=" * 50)
    
    from model import MultiTaskTransformer
    
    # 初始化模型
    model = MultiTaskTransformer(model_config)
    model.eval()
    
    # 模拟输入数据
    batch_size = 1
    seq_len = 20  # 20周历史数据
    n_stocks = 300
    n_market_factors = len(data_config.regime_factors)
    n_stock_factors = len(data_config.stock_factors)
    
    market_factors = torch.randn(batch_size, seq_len, n_market_factors)
    stock_factors = torch.randn(batch_size, n_stocks, n_stock_factors)
    
    # 前向传播
    with torch.no_grad():
        regime_logits, ranking_scores = model(market_factors, stock_factors)
    
    # 解析结果
    regime_probs = torch.softmax(regime_logits, dim=1)
    regime_pred = regime_logits.argmax(dim=1).item()
    
    regime_names = ['熊市', '震荡', '牛市']
    print(f"\n市场状态预测：{regime_names[regime_pred]}")
    print(f"概率分布：熊市={regime_probs[0,0]:.2%}, "
          f"震荡={regime_probs[0,1]:.2%}, "
          f"牛市={regime_probs[0,2]:.2%}")
    
    # 个股排序
    scores = ranking_scores.squeeze(0).numpy()
    top_10_indices = np.argsort(scores)[-10:][::-1]
    
    print(f"\nTop 10 股票索引：{top_10_indices}")
    print(f"对应预测分数：{scores[top_10_indices]}")


# ============================================================
# 示例4：回测指标计算
# ============================================================

def example_backtest_metrics():
    """示例：计算回测指标"""
    print("\n" + "=" * 50)
    print("示例4：回测指标计算")
    print("=" * 50)
    
    # 模拟策略收益率
    np.random.seed(42)
    n_days = 252 * 3  # 3年
    daily_returns = np.random.randn(n_days) * 0.015 + 0.0005  # 年化约12%收益
    
    # 计算指标
    total_return = (1 + daily_returns).prod() - 1
    annual_return = (1 + total_return) ** (1/3) - 1
    annual_volatility = daily_returns.std() * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 夏普比率
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # 卡尔玛比率
    calmar_ratio = annual_return / abs(max_drawdown)
    
    print(f"\n总收益率：{total_return:.2%}")
    print(f"年化收益率：{annual_return:.2%}")
    print(f"年化波动率：{annual_volatility:.2%}")
    print(f"最大回撤：{max_drawdown:.2%}")
    print(f"夏普比率：{sharpe_ratio:.2f}")
    print(f"卡尔玛比率：{calmar_ratio:.2f}")


# ============================================================
# 示例5：因子IC分析
# ============================================================

def example_factor_ic_analysis():
    """示例：因子IC分析"""
    print("\n" + "=" * 50)
    print("示例5：因子IC分析")
    print("=" * 50)
    
    from scipy.stats import spearmanr
    
    # 模拟因子值和未来收益
    n_stocks = 300
    n_periods = 100
    
    ics = []
    
    for t in range(n_periods):
        # 模拟因子值（有一定预测能力）
        factor_values = np.random.randn(n_stocks)
        
        # 模拟未来收益（与因子正相关）
        noise = np.random.randn(n_stocks) * 0.8
        future_returns = 0.3 * factor_values + noise
        
        # 计算IC
        ic, _ = spearmanr(factor_values, future_returns)
        ics.append(ic)
    
    ics = np.array(ics)
    
    # 统计指标
    ic_mean = ics.mean()
    ic_std = ics.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    win_rate = (ics > 0).mean()
    
    print(f"\nIC均值：{ic_mean:.4f}")
    print(f"IC标准差：{ic_std:.4f}")
    print(f"IC_IR：{ic_ir:.4f}")
    print(f"胜率：{win_rate:.2%}")
    print(f"IC分布：min={ics.min():.4f}, max={ics.max():.4f}")
    
    # 判断因子有效性
    if ic_mean > 0.03 and ic_ir > 0.5 and win_rate > 0.5:
        print("\n✓ 因子有效！")
    else:
        print("\n✗ 因子可能无效，建议剔除")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    example_factor_calculation()
    example_data_preprocessing()
    example_model_inference()
    example_backtest_metrics()
    example_factor_ic_analysis()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()

