<div align="center">

# 多因子选股 | Transformer-MultiFactor-Stock

### Two-layer Transformer multi-factor stock selection.

Market-state recognition + stock ranking that fixes information leakage and boosts adaptivity.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**Transformer-MultiFactor-Stock** implements a **two-layer Transformer multi-factor stock-selection** strategy with two core modules — **market-state recognition** and **stock selection** — addressing the information-leakage and adaptivity problems of traditional multi-factor strategies, wrapped in a modular framework with backtesting.

> [!NOTE]
> 中文项目：Transformer 双层架构多因子选股——市场状态识别 + 个股选择，解决信息泄露，提升自适应与收益稳定性。

---

## Features

- **Two-layer Transformer** — market-state + stock-ranking modules.
- **Fixes information leakage** — cleaner factor modeling.
- **Adaptive** — handles market-style switches.
- **Modular framework** — data loader, factor library, model, backtest.

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/Transformer-MultiFactor-Stock.git
cd Transformer-MultiFactor-Stock

pip install -r strategy_framework/requirements.txt

python strategy_framework/example_usage.py   # usage
python strategy_framework/main.py            # run the strategy
```

Standalone timing / ranking scripts included at repo root.

---

## Project Structure

```
Transformer-MultiFactor-Stock/
├── strategy_framework/
│   ├── model.py, factor_library.py, data_loader.py
│   ├── backtest.py, config.py, main.py, example_usage.py
├── Transformer择时(1).py
└── Transformer编码器排序选股策略(1).py
```

---

## License

MIT — free to use, modify and distribute.
