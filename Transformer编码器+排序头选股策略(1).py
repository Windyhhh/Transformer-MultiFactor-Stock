# 股票池沪深300成分股（由于数据经过预处理，可能只有200只左右，存在excel表格里）
# 用于调试：训练集2020.01.01-2023.12.31；测试集：2024.01.01-2025.05.29
# 数据分布：第一列是股票代码，列名是Stock，第二列的日期，列名是date，第三、第四到最后的列都是因子，如factor1、factor2、factor3...
# 在排布上，先是第一只股票的全部日期数据，然后才是第二只股票的全部数据，往后一只一只堆叠


# 一、环境准备与数据处理
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 数据加载与预处理函数
def load_and_preprocess_data(file_path, factor_cols, return_col='return', date_col='date', stock_col='Stock',
                             standardize=False):
    """加载并预处理股票多因子数据"""
    # 加载Excel数据
    df = pd.read_excel(file_path)  # 在主函数里面传入了file_path具体路径

    # 确保日期列为datetime类型
    df[date_col] = pd.to_datetime(df[date_col])

    # 按股票和日期排序
    df = df.sort_values([stock_col, date_col])  # 先按股票代码分组，然后在每个股票代码组内，再按照日期进行升序排序，用于滑动窗口生成样本
    # 这样有个好处，哪怕后面新进来一些股票的新年份，也不需要按照面板数据的排版，只需要对齐收益率、开盘价、收盘价那些即可
    # 可以读取多个文件的！

    # 划分训练集和测试集（这里需要调参，后期可移到主函数里面）
    train_mask = df[date_col] <= '2023-12-31'
    test_mask = (df[date_col] >= '2024-01-01') & (df[date_col] <= '2025-05-29')

    train_df = df[train_mask].copy()  # 复制为副本，后续对副本数据进行修改时不会影响到原始数据
    test_df = df[test_mask].copy()

    # 数据标准化
    if standardize:  # 函数里面传入True表示需要标准化，传入False表示不需要进行标准化(如果excel里面已经标准化就不再标准化)
        scaler = StandardScaler()
        train_df[factor_cols] = scaler.fit_transform(train_df[factor_cols])
        test_df[factor_cols] = scaler.transform(test_df[factor_cols])
    else:
        scaler = None

    return train_df, test_df, scaler


# 构建Transformer编码器所需的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# 定义完整的Transformer编码器模型（添加排序头）
class TransformerEncoderModel(nn.Module):
    def __init__(self, num_factors,  # 这里需要调参
                 d_model,
                 nhead,
                 num_layers,
                 dim_feedforward,  # 前馈网络的中间层维度
                 dropout,  # Dropout概率，用于防止过拟合
                 output_dim):  # 模型最终输出的特征维度
        super(TransformerEncoderModel, self).__init__()

        # 嵌入层：将输入的多因子数据映射到 d_model 维度
        self.embedding = nn.Linear(num_factors, d_model)
        # 位置编码层：为输入序列添加位置信息
        self.positional_encoding = PositionalEncoding(d_model)
        # 定义单个Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # 堆叠多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # 全局平均池化层：将序列特征压缩为单个向量
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        # 全连接层：将特征维度降为 output_dim
        self.fc = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU()  # 保留激活函数
        )

        # 添加排序头（用于联合训练）
        self.ranking_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出排序得分
        )

    def forward(self, x):  # 前向传播
        # 嵌入到d_model维度
        x = self.embedding(x)
        # 添加位置编码
        x = self.positional_encoding(x)
        # 调整维度以适应Transformer输入格式 (seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        # 调整维度用于全局池化
        x = x.permute(1, 2, 0)
        # 全局池化
        x = self.global_pooling(x)
        # 展平
        x = x.flatten(1)
        # 全连接层降维到128维
        features = self.fc(x)
        # 添加排序得分输出
        scores = self.ranking_head(features).squeeze(-1)
        return features, scores


##### 二、数据加载与模型训练：
# 自定义数据集类（修改：按日期分组）
##### 二、数据加载与模型训练：
# 自定义数据集类（修复版）
class StockDataset(Dataset):
    def __init__(self, data, factor_cols, return_col, stock_col, date_col, time_steps=20):
        self.data = data
        self.factor_cols = factor_cols
        self.return_col = return_col
        self.stock_col = stock_col
        self.date_col = date_col
        self.time_steps = time_steps

        # 按股票分组
        self.stock_groups = data.groupby(stock_col)

        # 准备数据
        self.samples = self._prepare_samples()

        # 按日期组织样本
        self.date_to_indices = self._organize_by_date()
        self.dates = sorted(self.date_to_indices.keys())
        self.numeric_dates = [(date - pd.Timestamp('2000-01-01')).days for date in self.dates]

    def _prepare_samples(self):
        samples = []
        for stock_id, group in self.stock_groups:
            group = group.sort_values(self.date_col)
            if len(group) < self.time_steps:
                continue

            # 滑动窗口生成样本
            for i in range(len(group) - self.time_steps + 1):
                window = group.iloc[i:i + self.time_steps]
                features = window[self.factor_cols].values
                return_value = window.iloc[-1][self.return_col]
                date_value = window.iloc[-1][self.date_col]

                samples.append({
                    'stock_id': stock_id,
                    'date': date_value,
                    'features': features,
                    'return': return_value
                })
        return samples

    def _organize_by_date(self):
        date_dict = {}
        for idx, sample in enumerate(self.samples):
            date = sample['date']
            if date not in date_dict:
                date_dict[date] = []
            date_dict[date].append(idx)
        return date_dict

    def __len__(self):
        return len(self.dates)  # 日期数量

    def __getitem__(self, idx):
        date = self.dates[idx]
        sample_indices = self.date_to_indices[date]
        # 收集该日期所有股票的数据
        features_list = []
        returns_list = []
        stock_ids = []
        for sample_idx in sample_indices:
            sample = self.samples[sample_idx]
            features_list.append(sample['features'])
            returns_list.append(sample['return'])
            stock_ids.append(sample['stock_id'])
        return {
            'numeric_date': torch.tensor((date - pd.Timestamp('2000-01-01')).days, dtype=torch.long),
            'stock_features': torch.FloatTensor(np.array(features_list)),
            'stock_returns': torch.FloatTensor(np.array(returns_list)),
            'stock_ids': stock_ids
        }

# ListNet损失函数（新增）
def listnet_loss(y_pred, y_true):
    """
    ListNet损失函数
    y_pred: 模型预测的排序得分 [batch_size, n_stocks]
    y_true: 真实收益率 [batch_size, n_stocks]
    """
    # 计算预测得分的概率分布
    pred_probs = torch.softmax(y_pred, dim=-1)

    # 计算真实收益率的概率分布
    true_probs = torch.softmax(y_true, dim=-1)

    # 计算交叉熵损失
    loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-8), dim=-1)
    return loss.mean()


# 训练Transformer模型（联合训练）
def train_joint_model(train_dataset, val_dataset, num_factors, d_model, nhead, num_layers,
                      batch_size, learning_rate, num_epochs, dim_feedforward, dropout,
                      output_dim, device):
    """训练Transformer编码器模型（联合训练排序任务）"""
    # 创建自定义DataLoader
    train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True)  # batch_size=1 因为每个样本是一个日期
    val_loader = DataLoader(val_dataset, batch_size=28)

    # 初始化模型
    model = TransformerEncoderModel(
        num_factors=num_factors,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model = model.to(device)

    print("开始训练Transformer和排序模型（联合训练模式）...")

    # 存储训练损失历史
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        # 训练循环（按日期滚动训练）
        for batch in train_loader:
            # 获取数据
            stock_features = batch['stock_features'][0].to(device)  # [n_stocks, 20, n_factors]
            stock_returns = batch['stock_returns'][0].to(device)  # [n_stocks]

            # 前向传播
            _, scores = model(stock_features)  # 输出排序得分 [n_stocks]

            # 计算ListNet损失
            loss = listnet_loss(scores, stock_returns)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        train_loss_history.append(avg_train_loss)

        # 验证循环
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                stock_features = batch['stock_features'][0].to(device)
                stock_returns = batch['stock_returns'][0].to(device)

                _, scores = model(stock_features)
                loss = listnet_loss(scores, stock_returns)

                epoch_val_loss += loss.item()
                val_batches += 1

        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0
        val_loss_history.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("训练损失曲线已保存为'loss_curve.png'")

    return model


# 实现NDCG计算函数
def ndcg_score(y_true, y_score, k=10):
    """计算归一化折损累积增益(NDCG)"""
    # 按得分排序的索引
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order][:k]

    # 计算DCG
    dcg = np.sum((2 ** y_true_sorted - 1) / np.log2(np.arange(2, len(y_true_sorted) + 2)))

    # 计算IDCG（理想DCG）
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal_sorted = y_true[ideal_order][:k]
    idcg = np.sum((2 ** y_true_ideal_sorted - 1) / np.log2(np.arange(2, len(y_true_ideal_sorted) + 2)))

    # 返回NDCG
    return dcg / idcg if idcg > 0 else 0



#### 三、策略回测与可视化：
# 实现选股策略（修改：适应新的数据集结构）
def run_stock_selection_strategy(transformer_model, test_dataset, factor_cols,
                                 stock_col, date_col, n_stocks=10, device='cpu'):
    """运行选股策略并回测"""
    transformer_model.eval()

    # 策略回测结果
    daily_returns = []      #这是用于记录所有日期每个日期的投资组合日收益率的，是一个列表
    portfolio_values = [100000000.0]  # 初始资金为1亿
    selected_stocks_history = []
    daily_ndcg = []
    dates = test_dataset.dates

    print("开始策略回测...")
    for i, date in enumerate(dates):
        # 获取当前日期的数据
        date_data = test_dataset[i]
        stock_features = date_data['stock_features'].to(device)  # [n_stocks, 20, n_factors]
        stock_returns = date_data['stock_returns'].cpu().numpy()  # [n_stocks]

        # 获取该日期的所有股票代码
        date_mask = (test_dataset.data[date_col] == date)
        stock_ids = date_data['stock_ids']  # 直接从数据集样本获取

        # 如果股票数量不足，跳过
        if len(stock_ids) < n_stocks:
            print(f"警告：日期 {date} 的股票数量不足 {n_stocks}，跳过")
            daily_returns.append(0.0)
            portfolio_values.append(portfolio_values[-1])
            daily_ndcg.append((date, 0))
            continue

        # 使用Transformer提取特征和排序分
        with torch.no_grad():
            _, scores = transformer_model(stock_features)
            scores = scores.cpu().numpy()

        # 创建股票-预测得分映射
        stock_scores = {stock_ids[j]: scores[j] for j in range(len(stock_ids))}

        # 按得分排序
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)

        # 选择前n只股票
        selected_stocks = [s[0] for s in sorted_stocks[:n_stocks]]
        selected_stocks_history.append((date, selected_stocks))

        # 计算NDCG
        true_returns = np.array([stock_returns[j] for j, stock in enumerate(stock_ids) if stock in selected_stocks])
        pred_scores = np.array([stock_scores[stock] for stock in selected_stocks])

        if len(true_returns) > 0:
            ndcg = ndcg_score(true_returns, pred_scores, k=len(selected_stocks))
            daily_ndcg.append((date, ndcg))
        else:
            daily_ndcg.append((date, 0))

        #计算当日投资组合收益
        #查找当日所有股票及其对应收益率
        date_stock_returns = {stock_ids[j]: stock_returns[j] for j in range(len(stock_ids))}

        # 等权重投资
        weight = 1.0 / n_stocks if n_stocks > 0 else 0
        daily_return = 0.0

        for stock in selected_stocks:
            if stock in date_stock_returns:
                daily_return += weight * date_stock_returns[stock]    #这里调仓逻辑优点问题，后续需要改

        daily_returns.append(daily_return)

        # 计算投资组合价值
        portfolio_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(portfolio_value)

        # 打印进度
        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            print(f"Processing {i + 1}/{len(dates)} days, Portfolio Value: {portfolio_value:.4f}, NDCG: {ndcg:.4f}")

    # 计算累计收益
    cumulative_returns = [(1 + r) for r in daily_returns]   #这里只是初始化为“1+”的形式
    for i in range(1, len(cumulative_returns)):
        cumulative_returns[i] *= cumulative_returns[i - 1]   #这里也是“1+”的形式


    # 计算策略指标
    # 转换为百分比形式（减去1后乘以100）
    cumulative_percentage = [(r - 1) * 100 for r in cumulative_returns]
    annual_return = (1 + cumulative_percentage[-1]/100) ** (252 / len(daily_returns)) - 1
    sharpe_ratio = np.mean(daily_returns) * np.sqrt(252) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0

    # 计算最大回撤
    portfolio_values = np.array(portfolio_values[1:])  # 去掉初始值
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = np.min(drawdown)

    print(f"\n策略回测结果:")
    print(f"策略累计收益率: {cumulative_percentage[-1]:.2f}%")
    print(f"年化收益率: {annual_return * 100:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")

    return {
        'dates': dates,
        'daily_returns': daily_returns,
        'portfolio_values': portfolio_values,
        'selected_stocks': selected_stocks_history,
        'ndcg': daily_ndcg,
        'cumulative_percentage': cumulative_percentage,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }



# 可视化策略收益
def visualize_strategy_performance(results, hs300_file_path, title="多因子选股策略回测结果"):
    """可视化策略收益"""
    dates = results['dates']
    daily_returns = results['daily_returns']  # 新增：提取每日收益率数据
    ndcg = results['ndcg']
    cumulative_percentage = results['cumulative_percentage']     #提取每日累计收益率

    # 确保日期和每日收益率数量一致
    if len(dates) != len(daily_returns):
        print(f"警告：日期数量 {len(dates)} 与每日收益率数量 {len(daily_returns)} 不一致")
        dates = dates[:len(daily_returns)]
        daily_returns = daily_returns[:len(dates)]

    # 读取沪深300收益率数据
    hs300_df = pd.read_excel(hs300_file_path)
    hs300_df['Date'] = pd.to_datetime(hs300_df['Date'])
    hs300_df = hs300_df.set_index('Date')

    # 创建动态收益图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 绘制组合收益率曲线（修改：使用累计收益率数据并设置颜色为红色）
    # 设置曲线颜色（可随时修改）
    line_color = 'r'  # 策略收益率曲线颜色
    # 绘制组合收益率曲线（从0%开始）
    line1, = ax1.plot(dates, cumulative_percentage, color=line_color, label='Portfolio Return')

    # 绘制沪深300收益率曲线
    hs300_dates = [date for date in dates if date in hs300_df.index]
    hs300_returns = [hs300_df.loc[date, 'hs300_return'] * 100 for date in hs300_dates]
    line3, = ax1.plot(hs300_dates, hs300_returns, color='cornflowerblue', label='HS300 Return')

    ax1.set_ylabel('Portfolio Return (%)')
    ax1.yaxis.set_major_formatter(PercentFormatter())   #设置y轴为百分比格式
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # 绘制NDCG
    line2, = ax2.plot([d[0] for d in ndcg], [d[1] for d in ndcg], color='#FF9966', label='NDCG')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('NDCG')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 保存静态图
    plt.savefig('strategy_performance.png')

    print("策略收益图已保存为'strategy_performance.png'")

    return fig, (ax1, ax2)



# 主函数：整合所有步骤
def main():
    # 数据参数
    file_path = r"D:\PythonProject\Project01\数据获取与处理\merged_data01(处理版).xlsx"  # 替换为实际数据文件路径
    hs300_file_path = r"D:\PythonProject\Project01\数据获取与处理\hs300_index_returns.xlsx"  # 替换为实际沪深300数据文件路径

    # 根据实际Excel文件中的列名修改因子列名，注意大小写
    factor_cols = ['book_to_price_ratio', 'boll_down', 'natural_log_of_market_cap', 'momentum',
                   'earnings_to_price_ratio', 'earnings_yield', 'operating_assets']

    return_col = "return"
    date_col = "date"
    stock_col = "Stock"
    time_steps = 20  # 时间序列长度

    # 模型参数
    d_model = 512  # 是嵌入层self.embedding的输出维度，也是后续Transformer编码器层处理时每个时间步的特征维度。512、1024
    nhead = 8  # 多头注意力机制的头数：4、8、12、16
    num_layers = 6  # 编码器的层数3、4、5、6
    batch_size = 64  # 在训练模型时每次输入到模型中的样本数量32、64、128（这里的样本是时间步长为20天的每只股票的滑动形成的样本）
    learning_rate = 0.0001  # 优化器更新模型参数的步长，控制模型学习的速度。0.01\0.001\0.0001，看别人怎么选。
    num_epochs = 10  # 训练模型时整个数据集被遍历的次数，即模型训练轮次
    # 可以通过观察训练集和验证集的损失曲线来确定合适的 num_epochs。当验证集损失不再下降甚至开始上升时，
    # 说明模型可能已经过拟合，此时可以停止训练。一般可以先设置一个较大的 num_epochs，然后使用早停策略来提前终止训练。
    dim_feedforward = 2048  # 前馈网络的中间层维度
    dropout = 0.1  # Dropout概率，用于防止过拟合
    output_dim = 128  # transformer编码器最终输出的特征维度

    # 策略参数
    n_stocks = 20  # 每次选择的股票数量

    # 一、加载和预处理数据
    train_df, test_df, scaler = load_and_preprocess_data(file_path, factor_cols, return_col, date_col, stock_col)

    # 二、创建训练集和测试集（使用新的数据集类）
    train_dataset = StockDataset(train_df, factor_cols, return_col, stock_col, date_col, time_steps)
    test_dataset = StockDataset(test_df, factor_cols, return_col, stock_col, date_col, time_steps)

    # 打印数据集信息
    print(f"训练集日期数量: {len(train_dataset)}")
    print(f"测试集日期数量: {len(test_dataset)}")

    # 示例：查看第一个训练样本
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        # 通过numeric_date还原日期
        sample_date = pd.Timestamp('2000-01-01') + pd.Timedelta(days=sample['numeric_date'].item())
        print(f"样本日期: {sample_date}")
        print(f"股票数量: {sample['stock_features'].shape[0]}")
        print(f"特征维度: {sample['stock_features'].shape[1:]}")
    else:
        print("警告：训练集为空，请检查数据加载！")

    # 三、按时间顺序将训练集进一步划分为训练集和验证集
    # 由于新的数据集按日期组织，我们需要按时间顺序划分
    total_dates = len(train_dataset)
    train_size = int(0.8 * total_dates)

    # 创建索引列表
    indices = list(range(total_dates))

    # 划分训练集和验证集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # 四、训练Transformer模型（联合训练）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_model = train_joint_model(
        train_subset,
        val_subset,
        len(factor_cols),
        d_model,
        nhead,
        num_layers,
        batch_size,
        learning_rate,
        num_epochs,
        dim_feedforward,
        dropout,
        output_dim,
        device=device
    )


    # 七、运行选股策略
    results = run_stock_selection_strategy(
        transformer_model,
        test_dataset,
        factor_cols,
        stock_col,
        date_col,
        n_stocks,
        device
    )

    # 八、可视化策略结果
    visualize_strategy_performance(results, hs300_file_path)


if __name__ == "__main__":
    main()