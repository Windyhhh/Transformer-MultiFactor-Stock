<div align="center">

# 📈 Transformer-MultiFactor-Stock

### Two-layer Transformer multi-factor stock selection.

Timing + stock ranking with a modular strategy framework and backtest.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**Transformer-MultiFactor-Stock** implements a **two-layer Transformer** for multi-factor stock selection — combining **market timing** and **stock ranking** inside a modular strategy framework with backtesting.

> [!NOTE]
> 中文项目：双层 Transformer 多因子选股——择时 + 选股排序，模块化策略框架 + 回测。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/Transformer-MultiFactor-Stock.git
cd Transformer-MultiFactor-Stock

# framework usage example
python strategy_framework/example_usage.py

# run the framework
python strategy_framework/main.py
```

Standalone timing / ranking scripts (`Transformer择时(1).py`, `Transformer编码器排序选股策略(1).py`) are also included.

---

## Features

- **Two-layer Transformer** — timing + ranking.
- **Modular framework** — factor library, data loader, model, backtest.
- **Backtest support** — evaluate strategies end-to-end.

---

## Project Structure

```
Transformer-MultiFactor-Stock/
├── strategy_framework/
│   ├── model.py, factor_library.py, data_loader.py
│   ├── backtest.py, config.py
│   ├── main.py, example_usage.py
│   └── requirements.txt
├── Transformer择时(1).py
└── Transformer编码器排序选股策略(1).py
```

---

## License

MIT — free to use, modify and distribute.
