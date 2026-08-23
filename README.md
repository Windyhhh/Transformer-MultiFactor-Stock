# 📈 Transformer Multi-Factor Stock Selection | Transformer 多因子选股策略

> **Dual-layer Transformer architecture for quantitative stock selection: market timing layer + stock ranking layer. Modular strategy framework with factor library, backtesting engine, and training pipeline.**
>
> 双层 Transformer 架构量化选股：市场择时层 + 股票排序层。模块化策略框架，包含因子库、回测引擎和训练流水线。

---

## 🌟 Why This Project? | 项目亮点

Traditional multi-factor stock selection relies on linear models (Fama-French) that struggle with nonlinear factor interactions. This project implements a **dual-layer Transformer architecture** for quantitative stock selection: a **market timing layer** that predicts market regimes, and a **stock ranking layer** that uses Transformer encoders with a ranking head to select top stocks. The project includes a complete **modular strategy framework** with a factor library, backtesting engine, data loader, model trainer, and example usage — everything needed to research and deploy Transformer-based quantitative strategies.

传统多因子选股依赖线性模型（Fama-French），难以捕捉非线性因子交互。本项目实现了**双层 Transformer 架构**用于量化选股：**市场择时层**预测市场状态，**股票排序层**使用 Transformer 编码器加排序头选择优质股票。项目包含完整的**模块化策略框架**，含因子库、回测引擎、数据加载器、模型训练器和示例用法——研究和部署基于 Transformer 的量化策略所需的一切。

| Feature | Details |
|---------|---------|
| **Architecture** | Dual-layer: Market Timing Transformer + Stock Ranking Transformer |
| **Timing Layer** | Transformer-based market regime prediction |
| **Selection Layer** | Transformer encoder + ranking head (ListNet / LambdaRank) |
| **Factor Library** | Built-in multi-factor library (value, momentum, quality, volatility) |
| **Backtesting** | Event-driven backtest engine with transaction costs |
| **Training** | Modular trainer with early stopping, LR scheduling |
| **Data Loader** | Support for CSV / database / API data sources |
| **Evaluation** | Sharpe ratio, max drawdown, win rate, annualized return |
| **Framework** | PyTorch + Pandas + NumPy |

---

## 🏗️ Architecture | 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   Market Data (Multi-Factor)                   │
│         OHLCV + Fundamental + Technical Factors                 │
│         (Value, Momentum, Quality, Volatility, Liquidity)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Factor Library & Processing                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  factor_library.py                                        │  │
│  │  • Value factors: PE, PB, PS, EV/EBITDA                  │  │
│  │  • Momentum: 1M/3M/6M/12M returns, RSI                  │  │
│  │  • Quality: ROE, ROA, gross margin, debt ratio           │  │
│  │  • Volatility: realized vol, downside deviation           │  │
│  │  • Liquidity: turnover, Amihud illiquidity                │  │
│  │  • Factor normalization: cross-sectional z-score          │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│  Layer 1: Market Timing   │     │  Layer 2: Stock Selection │
│  Transformer择时.py       │     │  Transformer编码器+排序头  │
│                           │     │                           │
│  Input: Market-wide       │     │  Input: Stock-level       │
│  factors + index data     │     │  multi-factor sequences   │
│                           │     │                           │
│  • Transformer encoder    │     │  • Transformer encoder    │
│  • Regime classification  │     │  • Multi-head attention   │
│  • Bull/Bear/Neutral      │     │  • Ranking head           │
│  • Position sizing signal │     │  • ListNet / LambdaRank   │
│                           │     │  • Top-K stock selection  │
└──────────────┬───────────┘     └──────────────┬────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Fusion & Execution                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Timing signal determines market exposure (0-100%)    │  │
│  │  • Selection ranking determines stock weights             │  │
│  │  • Combined: position = timing_signal × stock_score      │  │
│  │  • Rebalance: daily / weekly / monthly                    │  │
│  │  • Transaction costs: commission + slippage + impact      │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backtesting & Evaluation                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  backtest.py                                              │  │
│  │  • Event-driven backtest engine                           │  │
│  │  • Portfolio NAV tracking                                 │  │
│  │  • Metrics: Sharpe, Sortino, MaxDD, WinRate, CAGR       │  │
│  │  • Benchmark comparison (CSI 300 / S&P 500)             │  │
│  │  • Trade log & position history                           │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure | 项目结构

```
Transformer-MultiFactor-Stock/
├── Transformer择时.py                      # Layer 1: Market timing Transformer (19KB)
├── Transformer编码器+排序头选股策略.py     # Layer 2: Stock selection Transformer (26KB)
├── test_all.py                             # Comprehensive test suite
├── README_START_HERE.md                    # Quick start guide
├── 爆款博客.md                              # Technical blog (57KB)
├── 交付清单.md                              # Deliverable checklist
├── 代码测试报告与问题诊断.md                # Code test report & diagnosis
├── 创新点评估与实现指南.md                  # Innovation assessment & implementation guide
├── 多因子选股策略答疑文档.md                # Q&A documentation
├── 最终交付总结.md                          # Final delivery summary
├── 策略优化建议与代码修改指南.md            # Strategy optimization guide
├── 项目总结与快速导航.md                    # Project summary & navigation
├── P0核心错误修复报告.md                    # P0 bug fix report
└── strategy_framework/                     # Modular strategy framework
    ├── main.py                             # Main entry point
    ├── config.py                           # Configuration management
    ├── model.py                            # Transformer model definitions
    ├── trainer.py                          # Training pipeline
    ├── backtest.py                         # Backtesting engine
    ├── data_loader.py                      # Data loading & preprocessing
    ├── factor_library.py                   # Multi-factor library
    ├── example_usage.py                    # Example usage demonstration
    ├── requirements.txt                    # Python dependencies
    └── README.md                           # Framework documentation
```

---

## 🚀 Quick Start | 快速开始

### 1. Installation | 安装

```bash
cd strategy_framework
pip install -r requirements.txt
```

### 2. Configure | 配置

Edit `strategy_framework/config.py`:
```python
class StrategyConfig:
    # Data
    data_source = "csv"  # csv / database / api
    stock_universe = "CSI300"
    start_date = "2018-01-01"
    end_date = "2024-12-31"

    # Model
    transformer_layers = 4
    hidden_dim = 256
    num_heads = 8
    dropout = 0.1

    # Training
    epochs = 100
    batch_size = 64
    learning_rate = 1e-4
    early_stopping_patience = 10

    # Backtest
    initial_capital = 10000000
    commission_rate = 0.0003
    rebalance_freq = "weekly"
    top_k_stocks = 20
```

### 3. Run Example | 运行示例

```bash
python strategy_framework/example_usage.py
```

### 4. Run Tests | 运行测试

```bash
python test_all.py
```

### 5. Programmatic Usage | 编程式使用

```python
from strategy_framework.config import StrategyConfig
from strategy_framework.data_loader import DataLoader
from strategy_framework.model import TransformerStockSelector
from strategy_framework.trainer import StrategyTrainer
from strategy_framework.backtest import BacktestEngine

# Initialize
config = StrategyConfig()
data_loader = DataLoader(config)
model = TransformerStockSelector(config)
trainer = StrategyTrainer(model, config)

# Train
train_data, val_data = data_loader.load()
trainer.train(train_data, val_data)

# Backtest
engine = BacktestEngine(model, config)
results = engine.run()
print(f"Annual Return: {results['cagr']:.2%}")
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## 🔬 Dual-Layer Architecture | 双层架构详解

### Layer 1: Market Timing Transformer | 第一层：市场择时 Transformer

**Purpose**: Predict market regime (bull / bear / neutral) and determine optimal market exposure.

**Input**: Market-wide factors (index returns, volatility, breadth, sentiment, macro indicators).

**Architecture**:
- Transformer encoder captures temporal dependencies in market factors
- Classification head outputs regime probabilities
- Position sizing: `exposure = P(bull) × 1.0 + P(neutral) × 0.5 + P(bear) × 0.0`

**Output**: Market exposure signal (0.0 – 1.0).

### Layer 2: Stock Selection Transformer | 第二层：股票选股 Transformer

**Purpose**: Rank stocks based on multi-factor sequences and select top-K performers.

**Input**: Per-stock multi-factor time series (value, momentum, quality, volatility, liquidity).

**Architecture**:
- Transformer encoder processes factor sequences for each stock
- Multi-head attention captures cross-factor and cross-time interactions
- Ranking head (ListNet / LambdaRank loss) outputs stock scores
- Top-K stocks selected for portfolio

**Output**: Stock ranking scores → portfolio weights.

### Strategy Fusion | 策略融合

```
final_position[i] = timing_signal × stock_score[i] / Σ(stock_score[top_k])
```

The timing signal modulates overall exposure, while stock scores determine relative weights within the portfolio.

---

## 📊 Factor Library | 因子库

| Category | Factors | Description |
|----------|---------|-------------|
| **Value** | PE, PB, PS, EV/EBITDA, dividend yield | Cheapness relative to fundamentals |
| **Momentum** | 1M/3M/6M/12M returns, RSI, MACD | Price trend strength |
| **Quality** | ROE, ROA, gross margin, debt ratio, accruals | Financial health & profitability |
| **Volatility** | Realized vol, downside deviation, beta, idiosyncratic vol | Risk characteristics |
| **Liquidity** | Turnover ratio, Amihud illiquidity, dollar volume | Trading liquidity |
| **Size** | Market cap, log(size) | Company size |
| **Growth** | Revenue growth, earnings growth, EPS growth | Business growth rate |

All factors are normalized using **cross-sectional z-score** at each rebalance date.

---

## 📈 Evaluation Metrics | 评估指标

| Metric | Formula | Description |
|--------|---------|-------------|
| **CAGR** | (End/Start)^(1/years) - 1 | Compound Annual Growth Rate |
| **Sharpe** | (R_p - R_f) / σ_p | Risk-adjusted return (annualized) |
| **Sortino** | (R_p - R_f) / σ_downside | Downside risk-adjusted return |
| **Max Drawdown** | max((peak-trough)/peak) | Maximum peak-to-trough decline |
| **Win Rate** | winning_trades / total_trades | Percentage of profitable trades |
| **Calmar** | CAGR / MaxDrawdown | Return per unit of drawdown |
| **Turnover** | avg(traded_value / portfolio_value) | Portfolio turnover rate |
| **Excess Return** | R_p - R_benchmark | Alpha over benchmark |

---

## 📚 References | 参考文献

1. **Vaswani, A., et al.** (2017). *Attention is all you need.* NeurIPS.
2. **Fama, E. F., & French, K. R.** (2015). *A five-factor asset pricing model.* Journal of Financial Economics.
3. **Xia, L., et al.** (2008). *Listwise approach to learning to rank: theory and algorithm.* ICML.
4. **Burges, C. J.** (2010). *From ranknet to lambdarank to lambdamart: An overview.* Learning.
5. **Lim, B., & Zohren, S.** (2021). *Time-series forecasting with deep learning: a survey.* Philosophical Transactions of the Royal Society A.
6. **Zhang, X., et al.** (2024). *Deep learning in quantitative finance: A survey.* arXiv.

---

## ⚠️ Disclaimer | 免责声明

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

本项目**仅供教育和研究目的**，不构成任何投资建议。历史表现不代表未来收益。投资决策前请自行研究并咨询专业财务顾问。

---

## 📄 License | 许可证

MIT License — free to use, modify, and distribute for research purposes.

---

<div align="center">

**Built with 📈 for quantitative finance research**

[Report Bug](https://github.com/Windyhhh/Transformer-MultiFactor-Stock/issues) · [Request Feature](https://github.com/Windyhhh/Transformer-MultiFactor-Stock/issues)

</div>
