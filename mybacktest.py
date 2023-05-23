from math import sqrt
import pandas as pd
import numpy as np

from pykrx import stock, bond
import ccxt
import pyupbit

from math import sqrt
from scipy import stats, optimize 
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import time
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------- #
# 구한 팩터로부터 롱숏 시그널을 생성합니다
# n_positions: long/short할 개수
# long_min: long을 하기 위해서는 해당 값보다 커야 함
# short_max: short을 하기 위해서는 해당 값보다 작아야 함 
def generate_signal_using_factor(factor, n_positions, long_min=-0, short_max=0, long_short=True):
    if long_short:
        n_assets = len(factor.columns) # 전체 asset의 개수를 구함
        long_signal = ((factor
                       .where(factor > long_min)
                       .rank(axis=1, ascending=False) <= n_positions)
                       .astype(int))
        short_signal = ((factor
                        .where(factor < short_max)
                        .rank(axis=1, ascending=True) <= n_positions)
                        .astype(int))
        signal = long_signal.add(short_signal.mul(-1))
    else:
        n_assets = len(factor.columns) # 전체 asset의 개수를 구함
        long_signal = ((factor
                       .where(factor > long_min)
                       .rank(axis=1, ascending=False) <= n_positions)
                       .astype(int))
        signal = long_signal
    return signal

# 위에서 generate_signal_using_factor를 바탕으로 생성된 signal을 바탕으로 동일가중 포트폴리오 생성
# 1/n_positions만큼 long short을하는 포트폴리오를 만든다
# 이 때, long또는 short을 하는 자산의 개수가 n_min_positions보다 작으면 거래를하지 않는다.
def generate_equal_weight_portfolio_from_signal(signal, n_positions, n_min_positions, long_short=True):
    if long_short:
        long = (signal == 1).astype(int)
        short = (signal == -1).astype(int)
        n_long = long.sum(axis=1)
        n_short = short.sum(axis=1)
        long_weight = long / n_positions
        short_weight = short / n_positions * (-1)
        weight = long_weight + short_weight
        weight[(n_long < n_min_positions) | (n_short < n_min_positions)]=0
    else:
        long = (signal == 1).astype(int)
        n_long = long.sum(axis=1)
        long_weight = long / n_positions
        weight = long_weight
        weight[(n_long < n_min_positions)]=0
    return weight

def generate_cap_weight_portfolio_from_signal(signal, n_positions,n_min_positions, cap, long_short=True):
    if long_short:
        long = (signal == 1).astype(int)
        short = (signal == -1).astype(int)
        n_long, n_short = long.sum(axis=1), short.sum(axis=1)
        long_weight = (long * cap).div((long*cap).sum(axis=1), axis=0)
        short_weight = (short * cap).div((short * cap).sum(axis=1), axis=0)* (-1)
        weight = long_weight + short_weight
        weight[(n_long < n_min_positions) | (n_short < n_min_positions)]=0
    else:
        long = (signal == 1).astype(int)
        n_long = long.sum(axis=1)
        long_weight = (long * cap).div((long*cap).sum(axis=1), axis=0)
        weight = long_weight
        weight[(n_long < n_min_positions)] = 0
    return weight

def backtest_from_weights(price_df, weights, start, rebalance_dates, commission=0, tax=0, interest=0):
    cash, nav = start, start
    assets = {x:0 for x in price_df.columns}
    nav_ls = [nav]
    for day in weights.index.tolist():
        current_price = price_df.loc[day, :].to_dict()
        if day in rebalance_dates:
            new_weight = weights.loc[day, :]
            for ticker in new_weight.keys():
                weight = new_weight[ticker] # 주식 비중
                price = current_price[ticker] # 주식 구매 가격
                target_amount= int(nav * weight /price) # 주식 구매 수량
                now_amount = assets[ticker]
                amount = target_amount - now_amount
                cash -= (amount * price) # 주식 구매 가격 * 수량
                cash -= (abs(amount * price) * commission) # 수수료
                if amount < 0:
                    cash -= (abs(amount * price) * tax) # 증권거래세
                assets[ticker] = target_amount # 주식명과 수량을 기록
        cash = cash * (1 + interest) # 이자(복리 가정)
        nav = cash
        for ticker in assets.keys():
            if ticker in current_price.keys():
                nav += (assets[ticker] * current_price[ticker]) * (1-commission-tax) # 현재보유하고 있는 주식의 가치를 더해 줌
        if nav <= 0:
            print(day)
            print("Reached 0")
            break
        nav_ls.append(nav)
    idx = weights.index.tolist()
    idx = [(idx[0] - timedelta(days=1))] + idx
    nav = pd.Series(nav_ls, index=idx)
    daily_return = nav.pct_change().dropna()
    return daily_return

def backtest_from_factor(price_df, factor_df, n_positions, n_min_positions,
                     start, rebalance_dates, cap = None, long_short=True,
                    long_min=-0, short_max=0, commission=0, tax=0, interest=0):
    signal = generate_signal_using_factor(factor_df, n_positions, long_min, short_max,long_short)
    if cap is None:
        weights = generate_equal_weight_portfolio_from_signal(signal, n_positions, n_min_positions, long_short)
    else:
        weights = generate_cap_weight_portfolio_from_signal(signal, n_positions, n_min_positions, cap, long_short)
    daily_return = backtest_from_weights(price_df, weights, start, rebalance_dates, commission=0, tax=0, interest=0)
    return daily_return

# 팩터를 n분위수로 나눈후, 해당 분위수를 long only한 포트폴리오의 성과를 비교
# 즉, 팩터가 얼마나 유의미한 지를 살펴봄
def test_factor(n, price_df, factor_df, money,rebalance_dates, commission=0, tax=0, interest=0):
    quantile_dict = {}
    n_assets = len(price_df.columns)
    num = int(n_assets / n)
    rank_df = factor_df.rank(axis=1, ascending=True)
    for i in range(n):
        start, end = i*num, (i+1)*num
        quantile_df = ((rank_df > start) & (rank_df <= end)).astype(int)
        quantile_weights = generate_equal_weight_portfolio_from_signal(quantile_df, num, 0, long_short=False)
        quantile_daily_return = backtest_from_weights(price_df, quantile_weights, money,
                                                      rebalance_dates, commission, tax, interest)
        annualized_return = annualize_return(quantile_daily_return)
        annualized_std = annualize_std(quantile_daily_return)
        annualized_sharpe = annualize_sharpe(quantile_daily_return)
        mdd = MDD(quantile_daily_return)[0]
        quantile_dict[f'Q{i+1}'] = [annualized_return, annualized_std, annualized_sharpe, mdd]
    quantile_df = pd.DataFrame(quantile_dict)
    quantile_df.index = ['Annualized_return', 'Annualized_std', 'Annualized_sharpe', 'MDD']
    return quantile_df

# 팩터를 n분위수로 나눈후, 해당 분위수를 long only한 포트폴리오의 성과를 비교
# 즉, 팩터가 얼마나 유의미한 지를 살펴봄
# 위의 것을 시각화 함
def plot_factor_test_result(n, price_df, factor_df, money, rebalance_dates, commission=0, interest=0, tax=0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,16))
    quantile_df = test_factor(n, price_df, factor_df, money, rebalance_dates, commission, interest, tax)
    annualized_return = quantile_df.loc['Annualized_return', :]
    annualized_std = quantile_df.loc['Annualized_std', :]
    annualized_sharpe = quantile_df.loc['Annualized_sharpe', :]
    MDD = quantile_df.loc['MDD', :]
    
    components = [annualized_return, annualized_std, annualized_sharpe, MDD]
    colors = ['blue', 'yellow', 'green', 'orange']
    titles = ['Annualized Return', 'Annualized Std', 'Annualized Sharpe', 'MDD']
    locs = [(0,0), (0,1), (1,0), (1,1)]
    
    for component, color, title, loc in zip(components, colors, titles, locs):
        graph = axes[loc[0]][loc[1]]
        graph.bar(x=component.index, height=component, color=color)
        graph.axhline(y=component.mean(), color='black', linestyle='dashed')
        graph.set_title(title, fontsize=16)
        graph.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.show()
    
# ----------------------------------------------------------------------------------- #

# start_date와 end_date 사이에 영업일을 구하는 함수(한국 기준)
def get_workdays(start_date, end_date):
    dates = stock.get_market_ohlcv(start_date, end_date, "005930").index
    dates = pd.to_datetime(dates, format='%Y-%m-%d')
    return dates

# start_date와 end_date 사이에 특정 시기 별 가장 첫번째 영업일을 구하는 함수(한국 기준)
def get_first_workday(start_date, end_date, freq='M'):
    dates = stock.get_market_ohlcv(start_date, end_date, "005930").index
    dates = dates.to_frame("Dates").resample(freq).first()['날짜'].tolist()
    return dates

# start_date와 end_date 사이에 특정 간격을 기준으로 영업일을 구하는 함수(한국 기준)
def get_workday_by_period(start_date, end_date, period=5):
    dates = stock.get_market_ohlcv(start_date, end_date, "005930").index
    dates = dates[::period]
    return dates

# 일정한 주기(일 단위)로 리밸런싱을 하는 경우
def rebalance_weights_by_period(weights, period):
    for date in weights.index:
        if date not in weights[::period]:
            weights.loc[date, :] = np.nan
    weights = weights.fillna(method='ffill')
    weights.index = pd.to_datetime(weights.index, format='%Y-%m-%d')
    return weights

# 정해진 리밸런스일에 리밸런싱을 하는 경우
def rebalance_weights(weights, rebalance_date):
    for date in weights.index:
        if date not in rebalance_date:
            weights.loc[date, :] = np.nan
    weights = weights.fillna(method='ffill')
    weights.index = pd.to_datetime(weights.index, format='%Y-%m-%d')
    return weights

# datetime 인덱스를 object 인덱스로 변환
def change_datetime_to_str(idx, format='%Y-%m-%d'):
    return idx.strftime(format)

# object 인덱스를 datetime 인덱스로 변환
def change_str_to_datetime(idx, format='%Y-%m-%d'):
    return pd.to_datetime(idx, format)

# period 인덱스를 datetime 인덱스로 변환
def change_period_to_datetime(idx, format='%Y-%m-%d'):
    return idx.to_timestamp()

# ----------------------------------------------------------------------------------- #

# pykrx를 사용해서 원하는 기간동안 원하는 ticker의 가격 데이터를 가져옴
def return_price_df_pykrx(tickers, start_date, end_date):
    df = pd.DataFrame()
    for ticker in tqdm(tickers):
        price_df = stock.get_market_ohlcv(start_date, end_date, ticker)['종가'].to_frame(ticker)
        df = pd.concat([df, price_df],axis=1)
    return df

# binance를 사용해서 원하는 기간동안 ticker의 가격 데이터를 가져옴
# 제일 최근 500개의 데이터를 가져오기 때문에 없는 것들이 다수 있을 수 있음
def return_price_df_binance(tickers, start_date, end_date, unit='1d'):
    binance = ccxt.binance()
    df = pd.DataFrame()
    for ticker in tqdm(tickers):
        ohlcv = binance.fetch_ohlcv(ticker, unit)
        price_df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'vol'])
        price_df = price_df[['date','close']].set_index('date')
        price_df.columns = [ticker]
        price_df.index = pd.to_datetime(price_df.index, unit = 'ms')
        df = pd.concat([df, price_df],axis=1)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df

def return_price_df_pyupbit(tickers, start_date, end_date, unit='day'):
    df = pd.DataFrame()
    for ticker in tqdm(tickers):
        ohlcv = pyupbit.get_ohlcv(ticker, interval=unit)
        price_df = ohlcv[['close']]
        price_df.columns = [ticker]
        price_df.index = price_df.index.to_period("D").to_timestamp()
        df = pd.concat([df, price_df], axis=1)
        time.sleep(0.1)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df
                                      
# ----------------------------------------------------------------------------------- #
# 누적 수익률을 계산
def calculate_cumulative_return(daily_return):
    cumulative_return = np.exp(np.log(daily_return+1).cumsum())-1
    return cumulative_return

# 월별 수익률을 계산
def calculate_monthly_return(daily_return):
    cumulative_return = calculate_cumulative_return(daily_return)
    cumulative_return = cumulative_return.to_frame('cumulative_return')
    first_month_return = cumulative_return.resample("M").last().iloc[0, :]
    monthly_return = cumulative_return.resample("M").last().add(1).pct_change()
    monthly_return.iloc[0, :] = first_month_return
    return monthly_return.to_period('M')

# 분비별 수익률을 계산
def calculate_quarterly_return(daily_return):
    cumulative_return = calculate_cumulative_return(daily_return)
    cumulative_return = cumulative_return.to_frame('cumulative_return')
    first_quarter_return = cumulative_return.resample("Q").last().iloc[0, :]
    quarterly_return = cumulative_return.resample("Q").last().add(1).pct_change()
    quarterly_return.iloc[0, :] = first_quarter_return
    return quarterly_return.to_period("Q")

# 연간 수익률을 계산
def calculate_yearly_return(daily_return):
    cumulative_return = calculate_cumulative_return(daily_return)
    cumulative_return = cumulative_return.to_frame('cumulative_return')
    first_year_return = cumulative_return.resample("Y").last().iloc[0, :]
    annual_return = cumulative_return.resample("Y").last().add(1).pct_change()
    annual_return.iloc[0, :] = first_year_return
    return annual_return.to_period("Y")

# 누적 수익률을 통해 연간 수익률 계산(월간은 12, 분기는 4를 곱하면 됨)
def annualize_return(daily_return, year=252):
    days = len(daily_return)
    cumulative_return = calculate_cumulative_return(daily_return)
    return (1+cumulative_return[-1])** (year/days) - 1

# 일별 수익률을 통해 연간 수익률의 표준편차를 계산(월간은 12^0.5, 분기는 2를 곱하면 됨)
def annualize_std(daily_return, year=252):
    return daily_return.std() * sqrt(year)

# 연간 샤프비율 구함
def annualize_sharpe(daily_return, rf=0, year=252):
    return (daily_return.mean() * (year ** 0.5) - rf) / (daily_return.std())

# 가장 수익률이 낮은 n개 계산
def MDD(daily_return, n=1):
    cumulative_return = calculate_cumulative_return(daily_return)
    MDD = (cumulative_return.add(1)).div(np.maximum.accumulate(cumulative_return).add(1)).subtract(1)
    return MDD.sort_values(ascending=True).head(n)    

# 수익률이 정규분포를 따른다는 가정하에 연간 VAR를 구함
def VAR(daily_return, alpha=0.05):
    returns = annualize_return(daily_return)
    std = annualize_std(daily_return)
    return norm.ppf(alpha, returns, std)

# 수익률이 정규분포를 따른다는 가정하에 연간 ES를 구함
def C_VAR(daily_return, alpha=0.05):
    returns = annualize_return(daily_return)
    std = annualize_std(daily_return)
    CVAR = returns - std * norm.pdf(norm.ppf(1-alpha))/alpha
    return CVAR

# 적자확률 계산 (수익률이 정규분포를 따른다는 가정하에 지정한 target_min_return보다 낮을 확률)
def Shortfall_Prob(daily_return, min_return=0):
    returns = annualize_return(daily_return)
    std = annualize_std(daily_return)
    rv = stats.norm()
    shortfall_prob = rv.cdf((min_return - returns)/std)
    return shortfall_prob

# 위에서 구한 것들을 한 번에 표시
def summary_stats(daily_return, rf=0, n=1, alpha=0.05, min_return=0, year=252, benchmark=None):
    if benchmark is not None:
        value_dict = {}
        for i, returns in enumerate([daily_return, benchmark]):
            cumulative_return = calculate_cumulative_return(returns)[-1]
            annualized_return = annualize_return(returns, year)
            annualized_std = annualize_std(returns, year)
            annualized_sharpe = annualize_sharpe(returns, rf, year)
            mdd = MDD(returns, n)[0]
            Var = VAR(returns, alpha)
            C_Var = C_VAR(returns, alpha)
            shortfall_prob = Shortfall_Prob(returns, min_return)
            ls = [cumulative_return, annualized_return, annualized_std, annualized_sharpe,mdd, Var,
                  C_Var, shortfall_prob]
            ls = [f"{x:.2%}" for x in ls]
            value_dict[i] = ls
        summary_stats = pd.DataFrame(value_dict)
        summary_stats.columns = ['Portfolio', 'Benchmark']
        summary_stats.index = ['Cumulative Return', 'Annualized Return', 'Annualized Std',
                                            'Annualized Sharpe', 'MDD', 'Var', 'C_Var', 
                                            f'Shortfall Prob(min_return={min_return:.2%})']
    else:
        value_dict = {}
        cumulative_return = calculate_cumulative_return(daily_return)[-1]
        annualized_return = annualize_return(daily_return, year)
        annualized_std = annualize_std(daily_return, year)
        annualized_sharpe = annualize_sharpe(daily_return, rf, year)
        mdd = MDD(daily_return, n)[0]
        Var = VAR(daily_return, alpha)
        C_Var = C_VAR(daily_return, alpha)
        shortfall_prob = Shortfall_Prob(daily_return, min_return)
        ls = [cumulative_return, annualized_return, annualized_std, annualized_sharpe,mdd, Var,
              C_Var, shortfall_prob]
        ls = [f"{x:.2%}" for x in ls]
        value_dict["Portfolio"] = ls
        summary_stats = pd.DataFrame(value_dict,
                                     index=['Cumulative Return', 'Annualized Return', 'Annualized Std',
                                            'Annualized Sharpe', 'MDD', 'Var', 'C_Var', 
                                            f'Shortfall Prob(min_return={min_return:.2%})'])
    return summary_stats

# ----------------------------------------------------------------------------------- #
def plot_cumulative_return(daily_return, benchmark=None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    cumulative_return = calculate_cumulative_return(daily_return)
    if benchmark is not None:
        benchmark_cumulative_return = calculate_cumulative_return(benchmark)
        df = pd.DataFrame({'Portfolio': cumulative_return, 'Benchmark':benchmark_cumulative_return})
        ax.plot(df, label=['Portfolio', 'Benchmark'])
    else:
        ax.plot(cumulative_return, label=['Portfolio'])
    ax.set_title("CUMULATIVE RETURN", fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='dashed')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.show()
    
def plot_return_hist(returns):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.hist(returns)
    ax.set_title("Histogram of returns", fontsize=16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    plt.show()
    

def plot_return(daily_return, benchmark = None):
    fig, axes = plt.subplots(ncols = 3, figsize=(20,8))
    cumulative_return_p = calculate_cumulative_return(daily_return)
    if benchmark is not None:
        cumulative_return_b = calculate_cumulative_return(benchmark)
        df = pd.DataFrame({'Portfolio': cumulative_return_p, 'Benchmark':cumulative_return_b})
        axes[0].plot(df, label=['Portfolio', 'Benchmark'])
    else:
        axes[0].plot(cumulative_return_p, label=['Portfolio'])
    axes[0].set_title("CUMULATIVE RETURN", fontsize=16)
    axes[0].legend(loc='best', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='dashed')
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    axes[1].hist(daily_return, color='gold')
    axes[1].set_title("Daily Return Histogram", fontsize=16)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    
    monthly_return = calculate_monthly_return(daily_return)
    axes[2].hist(monthly_return, color='green')
    axes[2].set_title("Monthly Return Histogram", fontsize=16)
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    plt.show()
    
# 일별 수익률의 n일차 이동평균을 표시
def plot_rolling_return(daily_return, window=22):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    rolling_return = daily_return.rolling(window).mean().dropna()
    ax.plot(rolling_return, color='blue')
    ax.axhline(y=rolling_return.mean(), color='black', linestyle='dashed')
    ax.set_title(f"Rolling Return (Window={window})", fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()

# 일별 수익률의 n일차 이동표준편차를 표시
def plot_rolling_std(daily_return, window=22):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    rolling_std = daily_return.rolling(window).std().dropna()
    ax.plot(rolling_std, color='gold')
    ax.axhline(y=rolling_std.mean(), color='black', linestyle='dashed')
    ax.set_title(f"Rolling Std (Window={window})", fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()
    
# n일차 이동샤프비율을 표시
def plot_rolling_sharpe(daily_return, window=22, rf=0):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    rolling_mean = daily_return.rolling(window).mean().dropna()
    rolling_std = daily_return.rolling(window).std().dropna()
    rolling_sharpe = rolling_mean.subtract(rf).div(rolling_std)
    ax.axhline(y=rolling_sharpe.mean(), color='black', linestyle='dashed')
    ax.plot(rolling_sharpe, color='green')
    ax.set_title(f"Rolling Sharpe (Window={window})", fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()
    
# Rolling Variable들을 한번에 그림
def plot_rolling_variables(daily_return, window=22, rf=0):
    fig, axes = plt.subplots(ncols=3, figsize=(20, 8))
    rolling_return = daily_return.rolling(window).mean().dropna()
    rolling_std = daily_return.rolling(window).std().dropna()
    rolling_sharpe = (rolling_return.subtract(rf)).div(rolling_std)
    
    locs = [0,1,2]
    components = [rolling_return, rolling_std, rolling_sharpe]
    titles = [f"Rolling Return (Window={window})", f"Rolling STD (Window={window})", f"Rolling Sharpe (Window={window})"]
    for loc, component, title in zip(locs, components, titles):
        axes[loc].plot(component, color='orange')
        axes[loc].set_title(title,fontsize=16)
        axes[loc].axhline(y=component.mean(), color='black', linestyle='dashed')
        axes[loc].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y))) 
    plt.show()
   
# ----------------------------------------------------------------------------------- #
