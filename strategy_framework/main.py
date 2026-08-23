"""
主程序：策略执行入口
"""
import torch
import pandas as pd
import numpy as np
from config import data_config, model_config, training_config, backtest_config
from data_loader import FactorDataLoader, create_dataloaders
from model import MultiTaskTransformer
from trainer import Trainer
from backtest import Backtester
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    """训练模型"""
    print("=" * 50)
    print("开始训练模型")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(data_config)
    
    # 2. 初始化模型
    print("\n2. 初始化模型...")
    model = MultiTaskTransformer(model_config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 训练
    print("\n3. 开始训练...")
    trainer = Trainer(model, training_config)
    trainer.fit(train_loader, val_loader)
    
    # 4. 加载最佳模型
    print("\n4. 加载最佳模型...")
    trainer.load_checkpoint('best_model.pth')
    
    return trainer.model, test_loader


def run_backtest(model):
    """运行回测"""
    print("\n" + "=" * 50)
    print("开始回测")
    print("=" * 50)
    
    # 1. 加载测试数据
    print("\n1. 加载测试数据...")
    # TODO: 加载完整的测试数据（包含价格信息）
    test_data = pd.DataFrame()  # 占位符
    
    # 2. 运行回测
    print("\n2. 运行回测...")
    backtester = Backtester(model, backtest_config)
    backtest_results = backtester.run_backtest(test_data)
    
    # 3. 计算指标
    print("\n3. 计算回测指标...")
    # TODO: 加载基准数据（沪深300）
    benchmark_returns = None
    metrics = backtester.calculate_metrics(backtest_results, benchmark_returns)
    
    # 4. 打印结果
    print("\n" + "=" * 50)
    print("回测结果")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return backtest_results, metrics


def visualize_results(backtest_results: pd.DataFrame, metrics: dict):
    """可视化回测结果"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 净值曲线
    ax = axes[0, 0]
    ax.plot(backtest_results['date'], backtest_results['portfolio_value'])
    ax.set_title('Portfolio Value Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True)
    
    # 2. 日收益率分布
    ax = axes[0, 1]
    ax.hist(backtest_results['daily_return'], bins=50, edgecolor='black')
    ax.set_title('Daily Return Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    # 3. 回撤曲线
    ax = axes[1, 0]
    cumulative = (1 + backtest_results['daily_return']).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    ax.set_title('Drawdown')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    
    # 4. 仓位变化
    ax = axes[1, 1]
    ax.plot(backtest_results['date'], backtest_results['position'])
    ax.set_title('Position Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position')
    ax.grid(True)
    
    # 5. 市场状态分布
    ax = axes[2, 0]
    regime_counts = backtest_results['market_regime'].value_counts()
    ax.bar(['Bear', 'Neutral', 'Bull'], 
           [regime_counts.get(0, 0), regime_counts.get(1, 0), regime_counts.get(2, 0)])
    ax.set_title('Market Regime Distribution')
    ax.set_ylabel('Count')
    ax.grid(True)
    
    # 6. 累计收益对比（如果有基准）
    ax = axes[2, 1]
    cumulative_return = (1 + backtest_results['daily_return']).cumprod() - 1
    ax.plot(backtest_results['date'], cumulative_return, label='Strategy')
    # TODO: 添加基准收益
    ax.set_title('Cumulative Return')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300)
    print("\n可视化结果已保存至 backtest_results.png")


def rolling_retrain():
    """滚动训练方案"""
    print("\n" + "=" * 50)
    print("滚动训练模式")
    print("=" * 50)
    
    # 定义训练窗口
    train_windows = [
        ('2010-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2010-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
        ('2010-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2010-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ]
    
    all_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(train_windows):
        print(f"\n训练窗口 {i+1}: {train_start} - {train_end}")
        print(f"测试窗口 {i+1}: {test_start} - {test_end}")
        
        # 更新配置
        data_config.train_start = train_start
        data_config.train_end = train_end
        data_config.test_start = test_start
        data_config.test_end = test_end
        
        # 训练模型
        model, _ = train_model()
        
        # 回测
        backtest_results, metrics = run_backtest(model)
        all_results.append({
            'period': f"{test_start}_{test_end}",
            'metrics': metrics,
            'results': backtest_results
        })
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("滚动训练汇总结果")
    print("=" * 50)
    
    summary = pd.DataFrame([
        {'period': r['period'], **r['metrics']} 
        for r in all_results
    ])
    print(summary)
    summary.to_csv('rolling_results.csv', index=False)


def factor_analysis():
    """因子有效性分析"""
    print("\n" + "=" * 50)
    print("因子有效性分析")
    print("=" * 50)
    
    # TODO: 实现因子IC分析、分层回测等
    # 1. 计算每个因子的IC均值、IC_IR
    # 2. 因子相关性矩阵
    # 3. 因子分层回测
    # 4. 因子衰减分析
    pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多因子选股策略')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'backtest', 'rolling', 'factor_analysis'],
                       help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径（用于backtest模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        model, test_loader = train_model()
        
    elif args.mode == 'backtest':
        # 回测模式
        if args.checkpoint is None:
            print("错误：回测模式需要指定--checkpoint参数")
            return
        
        # 加载模型
        model = MultiTaskTransformer(model_config)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 运行回测
        backtest_results, metrics = run_backtest(model)
        
        # 可视化
        visualize_results(backtest_results, metrics)
        
    elif args.mode == 'rolling':
        # 滚动训练模式
        rolling_retrain()
        
    elif args.mode == 'factor_analysis':
        # 因子分析模式
        factor_analysis()


if __name__ == '__main__':
    main()

