# 📈 Transformer 多因子选股 | Transformer-based Multi-Factor Stock Selection

> **用 Transformer 捕捉因子间的时序依赖，从海量因子中自动挖掘 Alpha——量化选股的深度学习解法。**
>
> *Capture temporal dependencies among factors with Transformer, automatically mining Alpha from massive factors — deep learning for quantitative stock selection.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🤖 **Transformer 选股** | Transformer for Stocks | 自注意力机制捕捉因子间复杂交互 |
| 🧮 **多因子融合** | Multi-Factor Fusion | 价值、成长、动量、质量等因子自动加权 |
| ⏱️ **时序建模** | Temporal Modeling | 捕捉因子的时间演化和 regime 切换 |
| 🎯 **端到端** | End-to-End | 从原始因子到股票排名，一步到位 |
| 📊 **回测框架** | Backtesting | 完整回测，含换手率、夏普比率等指标 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-black?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)

---

## 📊 策略对比 | Strategy Comparison

| 方法 | 因子交互 | 时序建模 | 非线性 | 可解释性 |
|------|---------|---------|--------|---------|
| 线性多因子 | ❌ 线性 | ❌ 无 | ❌ 线性 | ✅ 强 |
| XGBoost | ✅ 树模型 | ❌ 弱 | ✅ 非线性 | 🟡 中 |
| LSTM | ❌ 弱 | ✅ 强 | ✅ 非线性 | 🟡 中 |
| **Transformer (本项目)** | ✅ 自注意力 | ✅ 强 | ✅ 非线性 | 🟡 中 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/Transformer-MultiFactor-Stock.git
cd Transformer-MultiFactor-Stock
pip install -r requirements.txt
python train.py --market A股 --start 2018-01-01 --end 2023-12-31
python backtest.py --model checkpoint.pt
```

---

## 📂 项目结构 | Project Structure

```
Transformer-MultiFactor-Stock/
├── train.py                   # 训练入口
├── backtest.py                # 回测入口
├── requirements.txt           # 依赖
├── models/
│   └── transformer.py         # Transformer 选股模型
├── factors/                   # 因子计算
│   ├── value.py               # 价值因子
│   ├── momentum.py            # 动量因子
│   ├── quality.py             # 质量因子
│   └── growth.py              # 成长因子
├── data/                      # 行情数据
├── backtesting/               # 回测引擎
└── results/                   # 回测结果
```

---

## 🔬 核心架构 | Core Architecture

### Transformer 选股模型 | Transformer Stock Selector

```
输入: [股票数 × 时间步 × 因子数]
  ↓
位置编码 (Positional Encoding)
  ↓
多头自注意力 (Multi-Head Self-Attention)  ← 捕捉因子间交互
  ↓
前馈网络 (Feed-Forward Network)
  ↓
时序聚合 (Temporal Aggregation)
  ↓
输出: 股票预期收益排名
```

### 因子体系 | Factor System

| 因子类别 | 代表因子 |
|---------|---------|
| 价值 | PE、PB、PS、股息率 |
| 动量 | 过去 N 月收益、相对强弱 |
| 质量 | ROE、毛利率、资产负债率 |
| 成长 | 营收增速、净利润增速 |
| 波动率 | 日收益率标准差、Beta |
| 流动性 | 换手率、成交额 |

---

## 🎯 应用场景 | Use Cases

- 📊 **量化投资**：构建多头选股策略，跑赢指数
- 🏦 **基金管理**：辅助基金经理进行股票筛选
- 📈 **指数增强**：在指数基础上进行选股增强
- 🎓 **学术研究**：深度学习在量化金融中的应用研究

---

## ⚠️ 风险提示 | Risk Disclaimer

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **量化 + AI 的完美结合，Star ⭐ 支持开源量化！**
