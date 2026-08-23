"""
因子库：常用因子计算函数
"""
import pandas as pd
import numpy as np
from typing import Union

class FactorCalculator:
    """因子计算器"""
    
    # ==================== 估值因子 ====================
    
    @staticmethod
    def pe_ttm(price: pd.Series, eps_ttm: pd.Series) -> pd.Series:
        """市盈率TTM"""
        return price / eps_ttm
    
    @staticmethod
    def pb(price: pd.Series, bps: pd.Series) -> pd.Series:
        """市净率"""
        return price / bps
    
    @staticmethod
    def ps_ttm(market_cap: pd.Series, revenue_ttm: pd.Series) -> pd.Series:
        """市销率TTM"""
        return market_cap / revenue_ttm
    
    @staticmethod
    def pcf_ttm(market_cap: pd.Series, cashflow_ttm: pd.Series) -> pd.Series:
        """市现率TTM"""
        return market_cap / cashflow_ttm
    
    @staticmethod
    def ev_ebitda(enterprise_value: pd.Series, ebitda: pd.Series) -> pd.Series:
        """企业价值倍数"""
        return enterprise_value / ebitda
    
    # ==================== 盈利因子 ====================
    
    @staticmethod
    def roe(net_profit: pd.Series, equity: pd.Series) -> pd.Series:
        """净资产收益率"""
        return net_profit / equity
    
    @staticmethod
    def roa(net_profit: pd.Series, total_assets: pd.Series) -> pd.Series:
        """总资产收益率"""
        return net_profit / total_assets
    
    @staticmethod
    def gross_margin(gross_profit: pd.Series, revenue: pd.Series) -> pd.Series:
        """毛利率"""
        return gross_profit / revenue
    
    @staticmethod
    def net_margin(net_profit: pd.Series, revenue: pd.Series) -> pd.Series:
        """净利率"""
        return net_profit / revenue
    
    @staticmethod
    def roic(nopat: pd.Series, invested_capital: pd.Series) -> pd.Series:
        """投入资本回报率"""
        return nopat / invested_capital
    
    # ==================== 成长因子 ====================
    
    @staticmethod
    def revenue_growth(revenue: pd.Series, periods: int = 4) -> pd.Series:
        """营收增长率（同比）"""
        return revenue.pct_change(periods)
    
    @staticmethod
    def profit_growth(net_profit: pd.Series, periods: int = 4) -> pd.Series:
        """净利润增长率（同比）"""
        return net_profit.pct_change(periods)
    
    @staticmethod
    def roe_growth(roe: pd.Series, periods: int = 4) -> pd.Series:
        """ROE增长率"""
        return roe.diff(periods)
    
    @staticmethod
    def eps_growth(eps: pd.Series, periods: int = 4) -> pd.Series:
        """EPS增长率"""
        return eps.pct_change(periods)
    
    # ==================== 质量因子 ====================
    
    @staticmethod
    def debt_to_asset(total_debt: pd.Series, total_assets: pd.Series) -> pd.Series:
        """资产负债率"""
        return total_debt / total_assets
    
    @staticmethod
    def current_ratio(current_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
        """流动比率"""
        return current_assets / current_liabilities
    
    @staticmethod
    def quick_ratio(current_assets: pd.Series, inventory: pd.Series, 
                   current_liabilities: pd.Series) -> pd.Series:
        """速动比率"""
        return (current_assets - inventory) / current_liabilities
    
    @staticmethod
    def receivable_turnover(revenue: pd.Series, receivables: pd.Series) -> pd.Series:
        """应收账款周转率"""
        return revenue / receivables
    
    @staticmethod
    def inventory_turnover(cogs: pd.Series, inventory: pd.Series) -> pd.Series:
        """存货周转率"""
        return cogs / inventory
    
    # ==================== 动量因子 ====================
    
    @staticmethod
    def momentum(prices: pd.Series, window: int = 20) -> pd.Series:
        """动量因子"""
        return prices.pct_change(window)
    
    @staticmethod
    def return_1m(prices: pd.Series) -> pd.Series:
        """过去1月收益率"""
        return prices.pct_change(20)  # 约20个交易日
    
    @staticmethod
    def return_3m(prices: pd.Series) -> pd.Series:
        """过去3月收益率"""
        return prices.pct_change(60)
    
    @staticmethod
    def return_6m(prices: pd.Series) -> pd.Series:
        """过去6月收益率"""
        return prices.pct_change(120)
    
    @staticmethod
    def return_12m(prices: pd.Series) -> pd.Series:
        """过去12月收益率"""
        return prices.pct_change(240)
    
    # ==================== 波动因子 ====================
    
    @staticmethod
    def volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """历史波动率"""
        return returns.rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def beta(stock_returns: pd.Series, market_returns: pd.Series, 
            window: int = 60) -> pd.Series:
        """Beta系数"""
        def calc_beta(x, y):
            covariance = np.cov(x, y)[0, 1]
            variance = np.var(y)
            return covariance / variance if variance > 0 else 0
        
        result = pd.Series(index=stock_returns.index, dtype=float)
        for i in range(window, len(stock_returns)):
            result.iloc[i] = calc_beta(
                stock_returns.iloc[i-window:i].values,
                market_returns.iloc[i-window:i].values
            )
        return result
    
    @staticmethod
    def downside_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """下行波动率"""
        def calc_downside_vol(x):
            negative_returns = x[x < 0]
            return negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        return returns.rolling(window).apply(calc_downside_vol)
    
    # ==================== 流动性因子 ====================
    
    @staticmethod
    def turnover(volume: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
        """换手率"""
        return volume / shares_outstanding
    
    @staticmethod
    def avg_turnover(volume: pd.Series, shares_outstanding: pd.Series, 
                    window: int = 20) -> pd.Series:
        """平均换手率"""
        daily_turnover = volume / shares_outstanding
        return daily_turnover.rolling(window).mean()
    
    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume_dollar: pd.Series, 
                          window: int = 20) -> pd.Series:
        """Amihud非流动性指标"""
        daily_illiq = returns.abs() / volume_dollar
        return daily_illiq.rolling(window).mean()
    
    # ==================== 技术因子 ====================
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI相对强弱指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2):
        """布林带"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return upper, ma, lower
    
    @staticmethod
    def ma_ratio(prices: pd.Series, short_window: int = 5, 
                long_window: int = 20) -> pd.Series:
        """均线比率"""
        ma_short = prices.rolling(short_window).mean()
        ma_long = prices.rolling(long_window).mean()
        return ma_short / ma_long
    
    # ==================== 另类因子 ====================
    
    @staticmethod
    def analyst_rating_change(ratings: pd.Series, window: int = 60) -> pd.Series:
        """分析师评级变化"""
        return ratings.diff(window)
    
    @staticmethod
    def analyst_coverage(num_analysts: pd.Series) -> pd.Series:
        """分析师覆盖度"""
        return num_analysts
    
    @staticmethod
    def earnings_surprise(actual_eps: pd.Series, expected_eps: pd.Series) -> pd.Series:
        """盈利超预期"""
        return (actual_eps - expected_eps) / expected_eps.abs()

