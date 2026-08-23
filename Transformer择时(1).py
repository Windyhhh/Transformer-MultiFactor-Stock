# 沪深300指数择时模型（基于完整Transformer）
# 用于调试：训练集2020.01.01-2023.12.31；测试集：2024.01.01-2025.05.29
# 数据格式：日期(date)、沪深300指数收盘价(close)及相关技术指标因子
# 目标：预测未来n天的沪深300指数走势，用于择时策略

# 一、环境准备与数据处理
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 数据加载与预处理函数
def load_and_preprocess_data(file_path, feature_cols, target_col='close', date_col='date',
                             standardize=True, predict_days=5):
    """加载并预处理沪深300指数数据"""
    # 加载Excel数据
    df = pd.read_excel(file_path)

    # 确保日期列为datetime类型
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)  # 按日期排序

    # 划分训练集和测试集
    train_mask = df[date_col] <= '2023-12-31'
    test_mask = (df[date_col] >= '2024-01-01') & (df[date_col] <= '2025-05-29')

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # 创建目标变量：未来n天的指数值
    train_df[f'future_{predict_days}_day'] = train_df[target_col].shift(-predict_days)
    test_df[f'future_{predict_days}_day'] = test_df[target_col].shift(-predict_days)

    # 移除最后n行（没有目标值的数据）
    train_df = train_df.iloc[:-predict_days]
    test_df = test_df.iloc[:-predict_days]

    # 数据标准化
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        # 对目标值单独标准化
        target_scaler = StandardScaler()
        train_target = train_df[[f'future_{predict_days}_day']]
        test_target = test_df[[f'future_{predict_days}_day']]
        train_df[[f'future_{predict_days}_day']] = target_scaler.fit_transform(train_target)
        test_df[[f'future_{predict_days}_day']] = target_scaler.transform(test_target)
        scaler.target_scaler = target_scaler

    return train_df, test_df, scaler


# 构建Transformer所需的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 定义完整的Transformer模型（编码器+解码器）
class TransformerModel(nn.Module):
    def __init__(self, input_dim,  # 输入特征维度
                 d_model,  # 模型维度
                 nhead,  # 多头注意力头数
                 num_encoder_layers,  # 编码器层数
                 num_decoder_layers,  # 解码器层数
                 dim_feedforward,  # 前馈网络维度
                 dropout=0.1,  # Dropout概率
                 output_dim=1):  # 输出维度（预测值）
        super(TransformerModel, self).__init__()

        # 输入层：将特征映射到d_model维度
        self.encoder_input = nn.Linear(input_dim, d_model)
        self.decoder_input = nn.Linear(output_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # 输出层：预测未来指数值
        self.output_layer = nn.Linear(d_model, output_dim)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 编码器部分
        src = self.encoder_input(src) * np.sqrt(self.d_model)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, src_mask)

        # 解码器部分
        tgt = self.decoder_input(tgt) * np.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)

        # 输出预测结果
        output = self.output_layer(output)
        return output


# 自定义数据集类
class IndexDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, date_col, seq_len=60, pred_len=5):
        self.data = data
        self.features = data[feature_cols].values
        self.targets = data[target_col].values
        self.dates = data[date_col].values
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 预测长度

        # 计算有效样本数量
        self.valid_indices = []
        for i in range(len(data) - seq_len - pred_len + 1):
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # 编码器输入：历史序列
        src = self.features[i:i + self.seq_len]

        # 解码器输入：目标序列的前n-1个值（用于教师强制）
        tgt = self.targets[i + self.seq_len:i + self.seq_len + self.pred_len - 1]

        # 目标值：未来序列
        target = self.targets[i + self.seq_len:i + self.seq_len + self.pred_len]

        # 对应的日期
        dates = self.dates[i + self.seq_len:i + self.seq_len + self.pred_len]

        return {
            'src': torch.FloatTensor(src),
            'tgt': torch.FloatTensor(tgt).unsqueeze(1),  # 增加特征维度
            'target': torch.FloatTensor(target),
            'dates': dates
        }


# 滚动训练函数
def rolling_train(model, train_data, feature_cols, target_col, date_col,
                  seq_len, pred_len, epochs_per_window, batch_size,
                  optimizer, criterion, device, window_size=252):
    """滚动训练模型"""
    # 记录所有窗口的训练损失
    all_train_loss = []

    # 计算总窗口数
    total_samples = len(train_data)
    num_windows = (total_samples - window_size) // (window_size // 2) + 1  # 50%重叠

    print(f"开始滚动训练，共{num_windows}个窗口...")

    for window_idx in range(num_windows):
        # 计算当前窗口的起始和结束索引
        start_idx = window_idx * (window_size // 2)
        end_idx = min(start_idx + window_size, total_samples)

        # 提取当前窗口数据
        window_data = train_data.iloc[start_idx:end_idx].copy()

        # 创建数据集和数据加载器
        dataset = IndexDataset(
            window_data, feature_cols, target_col, date_col,
            seq_len=seq_len, pred_len=pred_len
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练当前窗口
        model.train()
        window_loss = []

        for epoch in range(epochs_per_window):
            total_loss = 0.0
            batch_count = 0

            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                target = batch['target'].to(device)

                # 前向传播
                output = model(src.permute(1, 0, 2), tgt.permute(1, 0, 2))
                output = output.permute(1, 0, 2).squeeze(-1)

                # 计算损失
                loss = criterion(output, target)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            window_loss.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(
                    f"窗口 {window_idx + 1}/{num_windows},  epoch {epoch + 1}/{epochs_per_window}, 损失: {avg_loss:.6f}")

        all_train_loss.extend(window_loss)
        print(f"窗口 {window_idx + 1}/{num_windows} 训练完成，平均损失: {np.mean(window_loss):.6f}")

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Rolling Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('rolling_loss_curve.png')
    print("滚动训练损失曲线已保存为'rolling_loss_curve.png'")

    return model


# 预测函数
def predict(model, dataset, device, scaler=None, target_col='future_5_day'):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    actuals = []
    dates_list = []

    with torch.no_grad():
        for data in dataset:
            src = data['src'].unsqueeze(0).to(device)  # 增加批次维度
            tgt = data['tgt'].unsqueeze(0).to(device)

            # 预测
            output = model(src.permute(1, 0, 2), tgt.permute(1, 0, 2))
            output = output.permute(1, 0, 2).squeeze()

            # 存储预测结果和实际值
            pred = output.cpu().numpy()
            actual = data['target'].numpy()

            # 如果有标准化器，还原数据
            if scaler and hasattr(scaler, 'target_scaler'):
                pred = scaler.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                actual = scaler.target_scaler.inverse_transform(actual.reshape(-1, 1)).flatten()

            predictions.append(pred)
            actuals.append(actual)
            dates_list.append(data['dates'])

    return {
        'predictions': np.concatenate(predictions),
        'actuals': np.concatenate(actuals),
        'dates': np.concatenate(dates_list)
    }


# 实现择时策略回测
def run_timing_strategy(predictions, actuals, dates, initial_capital=100000000.0):
    """基于预测结果运行择时策略"""
    # 策略回测结果
    portfolio_values = [initial_capital]
    daily_returns = []
    positions = []  # 1表示持有，0表示空仓

    # 计算每日的预测方向和实际收益率
    for i in range(len(predictions) - 1):
        # 预测方向：上涨(1)或下跌(0)
        pred_change = 1 if predictions[i + 1] > predictions[i] else 0

        # 实际收益率
        actual_return = (actuals[i + 1] - actuals[i]) / actuals[i] if actuals[i] != 0 else 0

        # 策略：预测上涨则持有，否则空仓
        position = 1 if pred_change == 1 else 0
        positions.append(position)

        # 计算当日收益
        daily_return = position * actual_return
        daily_returns.append(daily_return)

        # 更新组合价值
        portfolio_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(portfolio_value)

    # 计算累计收益
    cumulative_returns = [(1 + r) for r in daily_returns]
    for i in range(1, len(cumulative_returns)):
        cumulative_returns[i] *= cumulative_returns[i - 1]

    # 计算策略指标
    cumulative_percentage = [(r - 1) * 100 for r in cumulative_returns]
    annual_return = (1 + cumulative_percentage[-1] / 100) ** (252 / len(daily_returns)) - 1 if daily_returns else 0
    sharpe_ratio = np.mean(daily_returns) * np.sqrt(252) / np.std(daily_returns) if np.std(
        daily_returns) > 0 and daily_returns else 0

    # 计算最大回撤
    portfolio_values_np = np.array(portfolio_values[1:])
    running_max = np.maximum.accumulate(portfolio_values_np)
    drawdown = (portfolio_values_np - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # 计算预测准确率
    correct = 0
    total = 0
    for i in range(len(predictions) - 1):
        pred_up = predictions[i + 1] > predictions[i]
        actual_up = actuals[i + 1] > actuals[i]
        if pred_up == actual_up:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0

    print(f"\n策略回测结果:")
    print(f"预测准确率: {accuracy:.2%}")
    print(f"策略累计收益率: {cumulative_percentage[-1]:.2f}%" if cumulative_percentage else "无数据")
    print(f"年化收益率: {annual_return * 100:.2f}%" if annual_return else "无数据")
    print(f"夏普比率: {sharpe_ratio:.2f}" if sharpe_ratio else "无数据")
    print(f"最大回撤: {max_drawdown * 100:.2f}%" if max_drawdown else "无数据")

    # 计算回归指标
    if len(predictions) > 0 and len(actuals) > 0:
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"平均绝对误差(MAE): {mae:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")

    return {
        'dates': dates,
        'predictions': predictions,
        'actuals': actuals,
        'daily_returns': daily_returns,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'cumulative_percentage': cumulative_percentage,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'accuracy': accuracy
    }


# 可视化策略结果
def visualize_strategy_performance(results, title="沪深300指数择时策略回测结果"):
    """可视化择时策略结果"""
    dates = results['dates']
    predictions = results['predictions']
    actuals = results['actuals']
    cumulative_percentage = results['cumulative_percentage']
    positions = results['positions']

    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    # 1. 绘制指数实际值与预测值
    ax1.plot(dates, actuals, label='实际指数值', color='blue')
    ax1.plot(dates, predictions, label='预测指数值', color='red', linestyle='--')
    ax1.set_title('沪深300指数实际值与预测值')
    ax1.set_ylabel('指数值')
    ax1.legend()
    ax1.grid(True)

    # 2. 绘制策略累计收益率
    ax2.plot(dates[:len(cumulative_percentage)], cumulative_percentage,
             label='策略累计收益率', color='green')
    ax2.set_title('策略累计收益率')
    ax2.set_ylabel('收益率 (%)')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.legend()
    ax2.grid(True)

    # 3. 绘制持仓情况
    ax3.step(dates[:len(positions)], positions, where='mid',
             label='持仓状态', color='purple')
    ax3.set_title('择时策略持仓状态')
    ax3.set_ylabel('持仓 (1=持有, 0=空仓)')
    ax3.set_xlabel('日期')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('timing_strategy_performance.png')
    print("择时策略表现图已保存为'timing_strategy_performance.png'")

    return fig, (ax1, ax2, ax3)


# 主函数：整合所有步骤
def main():
    # 数据参数
    file_path = r"D:\PythonProject\Project01\数据获取与处理\hs300_index_data.xlsx"  # 沪深300指数数据路径
    # 特征列名（根据实际数据调整）
    feature_cols = ['close', 'open', 'high', 'low', 'volume',
                    'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'kdj_k', 'kdj_d']
    target_col = 'close'  # 目标列：收盘价
    date_col = 'date'
    seq_len = 60  # 输入序列长度（使用过去60天数据）
    pred_len = 5  # 预测未来5天的指数

    # 模型参数
    d_model = 128  # 模型维度
    nhead = 4  # 多头注意力头数
    num_encoder_layers = 3  # 编码器层数
    num_decoder_layers = 3  # 解码器层数
    batch_size = 32  # 批次大小
    learning_rate = 0.0001  # 学习率
    epochs_per_window = 20  # 每个滚动窗口的训练轮次
    dim_feedforward = 512  # 前馈网络维度
    dropout = 0.2  # Dropout概率
    window_size = 504  # 滚动窗口大小（约2年交易日）

    # 一、加载和预处理数据
    train_df, test_df, scaler = load_and_preprocess_data(
        file_path, feature_cols, target_col, date_col,
        standardize=True, predict_days=pred_len
    )

    # 目标列名（由load_and_preprocess_data生成）
    future_target_col = f'future_{pred_len}_day'

    # 二、创建训练集和测试集
    train_dataset = IndexDataset(
        train_df, feature_cols, future_target_col, date_col,
        seq_len=seq_len, pred_len=pred_len
    )
    test_dataset = IndexDataset(
        test_df, feature_cols, future_target_col, date_col,
        seq_len=seq_len, pred_len=pred_len
    )

    # 打印数据集信息
    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"测试集样本数量: {len(test_dataset)}")

    # 三、初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = TransformerModel(
        input_dim=len(feature_cols),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    # 定义优化器和损失函数（使用MSE损失，股价预测常用）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 均方误差损失，股价预测领域常用

    # 四、滚动训练模型
    model = rolling_train(
        model, train_df, feature_cols, future_target_col, date_col,
        seq_len, pred_len, epochs_per_window, batch_size,
        optimizer, criterion, device, window_size
    )

    # 五、在测试集上预测
    test_results = predict(model, test_dataset, device, scaler, future_target_col)

    # 六、运行择时策略
    strategy_results = run_timing_strategy(
        test_results['predictions'],
        test_results['actuals'],
        test_results['dates']
    )

    # 七、可视化策略结果
    visualize_strategy_performance(strategy_results)


if __name__ == "__main__":
    main()