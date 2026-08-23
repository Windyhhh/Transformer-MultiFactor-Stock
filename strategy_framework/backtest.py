"""
回测模块：策略回测与性能评估
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch
from datetime import datetime

class Backtester:
    """回测引擎"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        
        # 回测结果记录
        self.portfolio_values = []
        self.positions = []
        self.trades = []
        self.daily_returns = []
    
    @torch.no_grad()
    def predict(self, market_factors: np.ndarray, 
                stock_factors: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        模型预测
        
        Args:
            market_factors: [seq_len, n_market_factors]
            stock_factors: [n_stocks, n_stock_factors]
            
        Returns:
            market_regime: 市场状态 (0=熊市, 1=震荡, 2=牛市)
            stock_scores: 个股预测分数 [n_stocks]
        """
        # 转换为tensor并添加batch维度
        market_tensor = torch.FloatTensor(market_factors).unsqueeze(0).to(self.device)
        stock_tensor = torch.FloatTensor(stock_factors).unsqueeze(0).to(self.device)
        
        # 预测
        regime_logits, ranking_scores = self.model(market_tensor, stock_tensor)
        
        # 解析结果
        market_regime = regime_logits.argmax(dim=1).item()
        stock_scores = ranking_scores.squeeze(0).cpu().numpy()
        
        return market_regime, stock_scores
    
    def construct_portfolio(self, stock_scores: np.ndarray, 
                           stock_codes: List[str],
                           market_regime: int) -> Dict[str, float]:
        """
        构建投资组合
        
        Args:
            stock_scores: 个股预测分数
            stock_codes: 股票代码列表
            market_regime: 市场状态
            
        Returns:
            portfolio: {stock_code: weight}
        """
        # 1. 根据分数排序，选取Top N
        top_n = self.config.top_n_stocks
        top_indices = np.argsort(stock_scores)[-top_n:]
        selected_stocks = [stock_codes[i] for i in top_indices]
        
        # 2. 确定总仓位（根据市场状态）
        if market_regime == 2:  # 牛市
            position = np.random.uniform(*self.config.position_bull)
        elif market_regime == 1:  # 震荡
            position = np.random.uniform(*self.config.position_neutral)
        else:  # 熊市
            position = np.random.uniform(*self.config.position_bear)
        
        # 3. 分配权重
        if self.config.weight_method == 'equal':
            # 等权重
            weight_per_stock = position / top_n
            portfolio = {code: weight_per_stock for code in selected_stocks}
        
        elif self.config.weight_method == 'risk_parity':
            # 风险平价（需要波动率数据）
            # TODO: 实现风险平价权重计算
            pass
        
        elif self.config.weight_method == 'optimized':
            # 均值-方差优化
            # TODO: 实现优化权重
            pass
        
        return portfolio
    
    def calculate_transaction_cost(self, old_portfolio: Dict[str, float],
                                   new_portfolio: Dict[str, float],
                                   portfolio_value: float) -> float:
        """
        计算交易成本
        
        Args:
            old_portfolio: 旧持仓
            new_portfolio: 新持仓
            portfolio_value: 组合总价值
            
        Returns:
            交易成本（金额）
        """
        # 计算换手金额
        turnover = 0
        all_stocks = set(old_portfolio.keys()) | set(new_portfolio.keys())
        
        for stock in all_stocks:
            old_weight = old_portfolio.get(stock, 0)
            new_weight = new_portfolio.get(stock, 0)
            turnover += abs(new_weight - old_weight)
        
        # 交易成本 = 换手金额 × (佣金率 + 滑点率)
        cost_rate = self.config.commission_rate + self.config.slippage_rate
        cost = turnover * portfolio_value * cost_rate
        
        return cost
    
    def run_backtest(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            test_data: 测试数据，包含因子和价格信息
            
        Returns:
            回测结果DataFrame
        """
        initial_capital = 1000000  # 初始资金100万
        portfolio_value = initial_capital
        current_portfolio = {}
        
        # 按交易日期分组
        trade_dates = test_data['trade_date'].unique()
        
        results = []
        
        for i, date in enumerate(trade_dates):
            print(f"Backtesting: {date}")
            
            # 获取当日数据
            day_data = test_data[test_data['trade_date'] == date]
            
            # 提取因子
            market_factors = self._extract_market_factors(test_data, date)
            stock_factors = self._extract_stock_factors(day_data)
            stock_codes = day_data['ts_code'].tolist()
            
            # 模型预测
            market_regime, stock_scores = self.predict(market_factors, stock_factors)
            
            # 构建新组合
            new_portfolio = self.construct_portfolio(stock_scores, stock_codes, market_regime)
            
            # 计算交易成本
            cost = self.calculate_transaction_cost(current_portfolio, new_portfolio, portfolio_value)
            portfolio_value -= cost
            
            # 更新持仓
            current_portfolio = new_portfolio
            
            # 计算当日收益（下一个交易日的收益）
            if i < len(trade_dates) - 1:
                next_date = trade_dates[i + 1]
                daily_return = self._calculate_portfolio_return(
                    current_portfolio, day_data, 
                    test_data[test_data['trade_date'] == next_date]
                )
                portfolio_value *= (1 + daily_return)
            else:
                daily_return = 0
            
            # 记录结果
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'market_regime': market_regime,
                'position': sum(new_portfolio.values()),
                'n_stocks': len(new_portfolio),
                'transaction_cost': cost
            })
        
        return pd.DataFrame(results)
    
    def _extract_market_factors(self, data: pd.DataFrame, 
                                current_date: str) -> np.ndarray:
        """提取市场因子时序数据"""
        print("警告: 使用模拟市场因子数据")
        # 返回 [seq_len, n_market_factors]
        from config import data_config
        seq_len = data_config.lookback_window
        n_factors = len(data_config.regime_factors)
        
        # 创建模拟的时序市场因子数据
        market_factors = np.random.randn(seq_len, n_factors)
        return market_factors
    
    def _extract_stock_factors(self, day_data: pd.DataFrame) -> np.ndarray:
        """提取个股因子"""
        print("警告: 使用模拟个股因子数据")
        # 返回 [n_stocks, n_stock_factors]
        from config import data_config
        n_stocks = len(day_data) if len(day_data) > 0 else 300
        n_factors = len(data_config.stock_factors)
        
        # 创建模拟的个股因子数据
        stock_factors = np.random.randn(n_stocks, n_factors)
        return stock_factors
    
    def _calculate_portfolio_return(self, portfolio: Dict[str, float],
                                   current_data: pd.DataFrame,
                                   next_data: pd.DataFrame) -> float:
        """计算组合收益率"""
        total_return = 0
        
        for stock_code, weight in portfolio.items():
            # 获取当前和下一期价格
            current_price = current_data[current_data['ts_code'] == stock_code]['close'].values
            next_price = next_data[next_data['ts_code'] == stock_code]['close'].values
            
            if len(current_price) > 0 and len(next_price) > 0:
                stock_return = (next_price[0] - current_price[0]) / current_price[0]
                total_return += weight * stock_return
        
        return total_return
    
    def calculate_metrics(self, backtest_results: pd.DataFrame,
                         benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        计算回测指标
        
        Args:
            backtest_results: 回测结果
            benchmark_returns: 基准收益率序列（沪深300）
            
        Returns:
            评估指标字典
        """
        returns = backtest_results['daily_return'].values
        portfolio_values = backtest_results['portfolio_value'].values
        
        # 收益指标
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        n_years = len(returns) / 252  # 假设252个交易日
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        
        # 风险指标
        annual_volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 风险调整收益
        risk_free_rate = self.config.risk_free_rate
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        # 如果有基准数据，计算超额收益和信息比率
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns.values
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            metrics['information_ratio'] = information_ratio
            metrics['excess_return'] = excess_returns.mean() * 252
        
        return metrics
