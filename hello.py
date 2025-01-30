# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. 데이터 수집 (Yahoo Finance) ---
def fetch_crypto_data(tickers, start_date='2023-01-01', end_date='2024-01-01'):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data.columns = [col.replace('-USD', '') for col in data.columns]
    return data.dropna()

# BTC와 AVAX 데이터 가져오기 (최근 1년)
tickers = 'BTC-USD AVAX-USD'
data = fetch_crypto_data(tickers, days=180)  # days: 최대 180일로 제한 (iOS 메모리 절약)

# --- 2. 헤지 비율 계산 (numpy 선형 회귀) ---
def calculate_hedge_ratio(y, x):
    X = np.vstack([np.ones(len(x)), x]).T
    return np.linalg.lstsq(X, y, rcond=None)[0][1]

hedge_ratio = calculate_hedge_ratio(data['BTC'], data['AVAX'])
print(f'계산된 헤지 비율: {hedge_ratio:.4f}')

# --- 3. 스프레드 & Z-Score 계산 ---
spread = data['BTC'] - hedge_ratio * data['AVAX']
z_score = (spread - spread.mean()) / spread.std()

# --- 4. 거래 신호 생성 ---
signals = pd.Series('Hold', index=data.index)
signals[z_score > 1.5] = 'Short Spread'  # BTC 매도 + AVAX 매수
signals[z_score < -1.5] = 'Long Spread'  # BTC 매수 + AVAX 매도
signals[z_score.abs() < 0.5] = 'Exit'    # 포지션 청산

# --- 5. 수익률 계산 (Carnets 최적화) ---
returns = pd.Series(0.0, index=data.index)
prev_prices = data.shift(1)  # 전일 가격

for i in range(1, len(data)):
    if signals[i-1] == 'Long Spread':
        ret = (data['BTC'][i] - prev_prices['BTC'][i]) - hedge_ratio * (data['AVAX'][i] - prev_prices['AVAX'][i])
    elif signals[i-1] == 'Short Spread':
        ret = hedge_ratio * (prev_prices['AVAX'][i] - data['AVAX'][i]) - (prev_prices['BTC'][i] - data['BTC'][i])
    else:
        ret = 0
    returns[i] = ret

cum_returns = (returns + 1).cumprod() - 1

# --- 6. 결과 시각화 (모바일 최적화) ---
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(z_score, label='Z-Score', linewidth=0.5)
plt.fill_between(z_score.index, 1.5, -1.5, color='red', alpha=0.1, label='Entry Zone')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(cum_returns * 100, label='Cumulative Return (%)', color='green', linewidth=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# --- 7. 최종 성능 리포트 ---
total_return = cum_returns[-1] * 100
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
print(f'''
[전략 리포트]
- 누적 수익률: {total_return:.2f}%
- 샤프 지수: {sharpe_ratio:.2f}
- 최대 포지션 기간: {len(signals[signals != "Hold"])}일
''')