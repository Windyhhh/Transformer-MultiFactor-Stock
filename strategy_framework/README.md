# 多因子选股策略框架

基于Transformer的多任务学习框架，同时完成市场状态识别和个股选择。

## 📁 项目结构

```
strategy_framework/
├── config.py              # 配置文件（数据、模型、训练、回测参数）
├── data_loader.py         # 数据加载与预处理
├── factor_library.py      # 因子计算库
├── model.py              # Transformer模型定义
├── trainer.py            # 训练模块
├── backtest.py           # 回测引擎
├── main.py               # 主程序入口
└── README.md             # 使用说明
```

## 🚀 快速开始

### 1. 环境配置

```bash
pip install torch pandas numpy tushare akshare scipy matplotlib seaborn
```

### 2. 数据准备

在 `data_loader.py` 中配置Tushare token：

```python
ts.set_token('your_tushare_token')
```

获取token：https://tushare.pro/register

### 3. 训练模型

```bash
python main.py --mode train
```

### 4. 运行回测

```bash
python main.py --mode backtest --checkpoint checkpoints/best_model.pth
```

### 5. 滚动训练

```bash
python main.py --mode rolling
```

## 📊 策略框架说明

### 双层架构

**第一层：市场状态识别**
- 输入：宏观因子 + 市场情绪因子 + 技术因子
- 模型：Transformer Encoder
- 输出：市场状态（牛市/震荡/熊市）+ 仓位建议

**第二层：个股选择**
- 输入：基本面因子 + 量价因子 + 另类因子
- 模型：Transformer Encoder + Ranking Head
- 输出：个股预期收益排序

### 创新点

1. **多任务学习**：单一模型同时完成择时和选股，共享特征表示
2. **因子动态加权**：根据市场状态自适应调整因子权重
3. **排序损失函数**：使用Pairwise Ranking Loss优化排序任务
4. **滚动训练**：避免前视偏差，模拟真实投资场景

## 🔧 配置说明

### 数据配置 (config.py)

```python
data_config = DataConfig(
    train_start="2010-01-01",
    train_end="2018-12-31",
    valid_start="2019-01-01",
    valid_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2024-12-31",
    lookback_window=20,  # 回看20周
    rebalance_freq="W-MON"  # 每周一调仓
)
```

### 模型配置

```python
model_config = ModelConfig(
    d_model=128,        # 嵌入维度
    n_heads=8,          # 注意力头数
    n_layers=4,         # Transformer层数
    d_ff=512,           # 前馈网络维度
    dropout=0.1,
    regime_classes=3,   # 市场状态数
)
```

### 回测配置

```python
backtest_config = BacktestConfig(
    top_n_stocks=30,              # 选取前30只股票
    weight_method="equal",        # 等权重
    position_bull=(0.8, 1.0),     # 牛市仓位
    position_neutral=(0.4, 0.6),  # 震荡仓位
    position_bear=(0.0, 0.2),     # 熊市仓位
    commission_rate=0.00025,      # 万2.5佣金
    slippage_rate=0.001,          # 0.1%滑点
)
```

## 📈 因子体系

### 第一套因子：市场状态识别（约35个）

**宏观因子（10-15个）**
- M2同比增速、社融增速、PMI、CPI、PPI
- 10年期国债收益率、信用利差

**市场情绪因子（10-15个）**
- 成交量变化率、换手率、涨跌停比
- 融资融券余额、北向资金流入、市场宽度

**技术因子（10个）**
- 指数MACD、RSI、布林带位置、均线系统

### 第二套因子：个股选择（约50个）

**基本面因子（15-20个）**
- 估值：PE、PB、PS、PCF、EV/EBITDA
- 盈利：ROE、ROA、毛利率、净利率
- 成长：营收增速、利润增速、ROE增速
- 质量：资产负债率、流动比率、周转率

**量价因子（15-20个）**
- 动量：1/3/6/12月收益率
- 反转：1周收益率
- 波动：波动率、Beta
- 流动性：换手率、Amihud非流动性

**另类因子（10-15个）**
- 分析师：一致预期变化、覆盖度
- 舆情：新闻情感得分（可选）

## 🔬 因子筛选流程

### 阶段1：初步筛选
- IC均值 > 0.03
- IC_IR > 0.5
- 胜率 > 50%

### 阶段2：相关性去重
- 层次聚类
- 相关性 > 0.7 的因子归为一组
- 每组选择IC_IR最高的因子

### 阶段3：稳定性检验
- 分年度IC检验
- 分行业IC检验
- 衰减测试

## 📊 回测指标

### 收益指标
- 年化收益率
- 累计收益率
- 超额收益（相对沪深300）

### 风险指标
- 最大回撤
- 年化波动率
- 下行波动率

### 风险调整收益
- 夏普比率
- 索提诺比率
- 卡尔玛比率
- 信息比率

## 🛠️ 后续开发建议

### Phase 1：数据准备（2周）
1. 实现 `data_loader.py` 中的数据获取函数
2. 连接Tushare/AkShare API
3. 实现因子计算pipeline
4. 数据清洗与预处理

### Phase 2：基线模型（2周）
1. 实现简单线性模型作为baseline
2. 验证数据pipeline正确性
3. 建立评估框架

### Phase 3：深度模型（3周）
1. 完善Transformer模型
2. 超参数调优
3. 滚动训练与回测

### Phase 4：创新点实现（2周）
1. 多任务学习优化
2. 因子动态加权机制
3. 消融实验

### Phase 5：论文撰写（2周）
1. 整理实验结果
2. 可视化分析
3. 撰写报告

## ⚠️ 重要注意事项

1. **避免未来函数**：使用T-1周数据预测T周，在T周一开盘执行
2. **幸存者偏差**：必须包含退市股票数据
3. **交易成本**：回测必须扣除手续费和冲击成本
4. **数据质量**：财务数据使用"报告期+45天"规则
5. **过拟合风险**：严格区分训练/验证/测试集

## 📚 参考资料

### 学术论文
- Barra Risk Model Handbook
- 101 Formulaic Alphas (WorldQuant)

### 数据平台
- Tushare: https://tushare.pro
- AkShare: https://akshare.akfamily.xyz
- 聚宽: https://www.joinquant.com

### 书籍
- Active Portfolio Management (Grinold & Kahn)
- 因子投资：方法与实践（石川）

## 📞 技术支持

如有问题，请参考《多因子选股策略答疑文档.md》

