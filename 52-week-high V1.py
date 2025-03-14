#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def compute_shares(spot, min_6m):
    """Compute number of shares to buy using the formula.
       With the new formula, shares = 200/(0.2*spot) = 1000/spot."""
    diff = 0.2 * spot
    if diff == 0:
        return 0
    return abs(200 / diff)  # which equals 1000/spot

def update_trailing_stop_new(entry_price, current_price, old_trailing_stop):
    """
    New trailing stop function:
    - At buy, stop = 0.8 * entry_price (i.e. -20% below entry).
    - Each day, candidate = 0.8 * current_price.
    - Trailing stop is updated as the maximum of its previous value and candidate.
    This way, the stop only moves upward.
    """
    candidate = 0.8 * current_price
    return max(old_trailing_stop, candidate)

def get_performance(current_price, reference_price):
    """Return performance as a fraction."""
    return (current_price - reference_price) / reference_price

# =============================================================================
# PARAMETERS AND SETTINGS
# =============================================================================
LOOKBACK_52W = 252      # trading days in 52 weeks
LOOKBACK_6M  = 126      # trading days in 6 months
REBALANCE_FREQ = 20     # used for checking new entries

PERF_UP_THRESHOLD   = 0.20   # +20% triggers an automatic re-buy for stocks already held
PERF_DOWN_THRESHOLD = -0.20  # -20% triggers a sale (via trailing stop)

VOLATILITY_THRESHOLD = 0.05  # weekly volatility (over last 6 months) must be below 10%
MIN_AVG_VOLUME       = 100000  # average daily volume > 100k shares
MIN_PRICE            = 1       # stock price must be > $1

# List of stock symbols for backtesting
symbols = ['AAPL', 'MSFT', 'GOOGL']  # (you can include as many as desired)
start_date = '2010-01-01'
end_date   = '2020-12-31'

# =============================================================================
# DATA DOWNLOAD (using yfinance)
# =============================================================================
data = {}
for symbol in symbols:
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        continue
    if 'Adj Close' in df.columns:
        df = df[['Adj Close', 'Volume']].rename(columns={'Adj Close': 'Close'})
    else:
        df = df[['Close', 'Volume']]
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)
    data[symbol] = df

# =============================================================================
# PORTFOLIO & TRADE LOG STRUCTURES
# =============================================================================
# Each symbol held in the portfolio is stored as:
#   - 'first_buy_date': date of the first buy (start of continuous position)
#   - 'accumulated_shares': total shares held
#   - 'monitoring': details of the most recent buy used for daily monitoring:
#         { 'entry_date', 'entry_price', 'base_stop', 'trailing_stop', 'signal_trigger', 'monitoring_shares' }
#   - 'all_buys': list of individual buy records (each with buy_date, buy_price, shares, signal_trigger)
portfolio = {}
# The trades list logs each trade event. (For re-buy events, a virtual trade record is logged.)
trades = []

# New list to log each order event (buy or sell) for cumulative investment tracking.
orders = []

# Global variable to track the cumulative net cash outlay.
# Each new buy costs $1,000 (subtract 1000) and each sale adds sale proceeds.
cumulative_investment = 0

# =============================================================================
# BACKTESTING LOOP
# =============================================================================
global_dates = list(next(iter(data.values())).index)

for i, current_date in enumerate(global_dates):
    # -------------------------------------------
    # DAILY UPDATE for stocks in portfolio:
    # -------------------------------------------
    for symbol in list(portfolio.keys()):
        df = data[symbol]
        if current_date not in df.index:
            continue
        current_idx = df.index.get_loc(current_date)
        current_price = float(df.iloc[current_idx]['Close'])
        pos = portfolio[symbol]
        mon = pos['monitoring']
        # Update trailing stop.
        mon['trailing_stop'] = update_trailing_stop_new(mon['entry_price'], current_price, mon['trailing_stop'])
        
        # Check stop loss: if current price falls below trailing stop, close entire position.
        if current_price < mon['trailing_stop']:
            # Before final sale, update any virtual re-buy trade records.
            for t in trades:
                if t['Symbol'] == symbol and t.get('virtual', False):
                    t['Stop Date'] = current_date
                    t['Spot at Stop'] = current_price
                    t['Reason for Stop'] = "stop loss hit"
                    t['virtual'] = False
            sale_proceeds = pos['accumulated_shares'] * current_price
            cumulative_investment += sale_proceeds
            # Log a sale order.
            orders.append({
                "Date": current_date,
                "Type": "sell",
                "Amount": sale_proceeds,
                "Cumulative Investment": cumulative_investment
            })
            trade = {
                'Symbol': symbol,
                'Entry Date': mon['entry_date'],  # using the most recent buy date
                'Stop Date': current_date,
                'Spot Entry': mon['entry_price'],
                'Spot at Stop': current_price,
                'Number of Shares': pos['accumulated_shares'],
                'Reason for Entry': mon['signal_trigger'],
                'Reason for Stop': "stop loss hit",
                'Cumulative Investment': cumulative_investment
            }
            trades.append(trade)
            del portfolio[symbol]
            continue  # move to next symbol
        
        # Check automatic re-buy condition: if performance from most recent buy exceeds +20%.
        perf = get_performance(current_price, mon['entry_price'])
        if perf > PERF_UP_THRESHOLD:
            virtual_trade = {
                'Symbol': symbol,
                'Entry Date': mon['entry_date'],
                'Stop Date': current_date,  # will be updated later upon final sale
                'Spot Entry': mon['entry_price'],
                'Spot at Stop': current_price,  # will be updated later
                'Number of Shares': mon['monitoring_shares'],
                'Reason for Entry': mon['signal_trigger'],
                'Reason for Stop': "stop loss hit",
                'virtual': True,
                'Cumulative Investment': cumulative_investment
            }
            trades.append(virtual_trade)
            last_6m = df.loc[:current_date].tail(LOOKBACK_6M)
            min_6m = float(last_6m['Close'].min())
            additional_shares = compute_shares(current_price, min_6m)
            # Each re-buy costs 1000.
            cost = 1000
            cumulative_investment -= cost
            orders.append({
                "Date": current_date,
                "Type": "buy",
                "Amount": cost,
                "Cumulative Investment": cumulative_investment
            })
            new_buy = {
                'buy_date': current_date,
                'buy_price': current_price,
                'shares': additional_shares,
                'signal_trigger': "+20% trigger"
            }
            pos['all_buys'].append(new_buy)
            pos['accumulated_shares'] = additional_shares
            # Update monitoring to new buy details.
            pos['monitoring'] = {
                'entry_date': current_date,
                'entry_price': current_price,
                'base_stop': min_6m,
                'trailing_stop': min_6m,
                'signal_trigger': "+20% trigger",
                'monitoring_shares': additional_shares
            }
    # -------------------------------------------
    # REBALANCE for new stock entries (only for stocks not in portfolio).
    # We check new entries every REBALANCE_FREQ trading days.
    # -------------------------------------------
    if i % REBALANCE_FREQ == 0:
        for symbol in symbols:
            if symbol not in data:
                continue
            if symbol in portfolio:
                continue
            df = data[symbol]
            if current_date not in df.index:
                continue
            current_idx = df.index.get_loc(current_date)
            if current_idx < 5 or current_idx < LOOKBACK_52W or current_idx < LOOKBACK_6M:
                continue
            last_week = df.iloc[current_idx-5:current_idx]
            last_year = df.iloc[max(0, current_idx-LOOKBACK_52W):current_idx]
            if float(last_week['Close'].max()) != float(last_year['Close'].max()):
                continue
            # Weekly volatility calculation: resample last 6 months to weekly data.
            last_6m = df.iloc[max(0, current_idx-LOOKBACK_6M):current_idx]
            weekly_prices = last_6m['Close'].resample('W').last()
            weekly_returns = weekly_prices.pct_change().dropna()
            weekly_volatility = float(weekly_returns.std())
            if weekly_volatility > VOLATILITY_THRESHOLD:
                continue
            avg_volume = float(last_week['Volume'].mean())
            if avg_volume < MIN_AVG_VOLUME:
                continue
            current_price = float(df.iloc[current_idx]['Close'])
            if current_price <= MIN_PRICE:
                continue
            min_6m_value = float(last_6m['Close'].min())
            shares = compute_shares(current_price, min_6m_value)
            cost = 1000
            cumulative_investment -= cost
            orders.append({
                "Date": current_date,
                "Type": "buy",
                "Amount": cost,
                "Cumulative Investment": cumulative_investment
            })
            portfolio[symbol] = {
                'first_buy_date': current_date,
                'accumulated_shares': shares,
                'monitoring': {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'base_stop': min_6m_value,
                    'trailing_stop': min_6m_value,
                    'signal_trigger': "52 week high",
                    'monitoring_shares': shares
                },
                'all_buys': [{
                    'buy_date': current_date,
                    'buy_price': current_price,
                    'shares': shares,
                    'signal_trigger': "52 week high"
                }]
            }
# -------------------------------------------
# FINALIZE: Close any open positions at the end of the backtest.
# -------------------------------------------
for symbol, pos in list(portfolio.items()):
    df = data[symbol]
    last_date = df.index[-1]
    last_price = float(df.iloc[-1]['Close'])
    mon = pos['monitoring']
    for t in trades:
        if t['Symbol'] == symbol and t.get('virtual', False):
            t['Stop Date'] = last_date
            t['Spot at Stop'] = last_price
            t['Reason for Stop'] = "stop loss hit"
            t['virtual'] = False
    sale_proceeds = pos['accumulated_shares'] * last_price
    cumulative_investment += sale_proceeds
    trade = {
        'Symbol': symbol,
        'Entry Date': mon['entry_date'],
        'Stop Date': last_date,
        'Spot Entry': mon['entry_price'],
        'Spot at Stop': last_price,
        'Number of Shares': pos['accumulated_shares'],
        'Reason for Entry': mon['signal_trigger'] if mon['signal_trigger'] == "52 week high" else "+20% on first trade",
        'Reason for Stop': "stop loss hit",
        'Cumulative Investment': cumulative_investment
    }
    trades.append(trade)
    del portfolio[symbol]

# =============================================================================
# EXPORT TO EXCEL
# =============================================================================
trades_df = pd.DataFrame(trades)
order_map = {symbol: i+1 for i, symbol in enumerate(symbols)}
trades_df['Order'] = trades_df['Symbol'].map(order_map)
trades_by_symbol = trades_df.sort_values(by=['Order', 'Entry Date']).drop(columns=['Order'])
trades_by_date = trades_df.sort_values(by=['Entry Date'])

# Add Trade Return and P&L columns.
for df_view in [trades_by_symbol, trades_by_date]:
    df_view["Trade Return"] = (df_view["Spot at Stop"] - df_view["Spot Entry"]) / df_view["Spot Entry"]
    df_view["P&L"] = df_view["Number of Shares"] * (df_view["Spot at Stop"] - df_view["Spot Entry"])

cols = ['Symbol', 'Entry Date', 'Stop Date', 'Spot Entry', 'Spot at Stop',
        'Number of Shares', 'Reason for Entry', 'Reason for Stop', 'Cumulative Investment',
        'Trade Return', 'P&L']
trades_by_symbol = trades_by_symbol[cols]
trades_by_date = trades_by_date[cols]

# Create Orders sheet from the orders list.
orders_df = pd.DataFrame(orders)
orders_df.sort_values(by="Date", inplace=True)

# Export to Excel with three sheets: Trades_by_Symbol, Trades_by_Date, and Orders.
excel_filename = "trade_log.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    trades_by_symbol.to_excel(writer, sheet_name="Trades_by_Symbol", index=False)
    trades_by_date.to_excel(writer, sheet_name="Trades_by_Date", index=False)
    orders_df.to_excel(writer, sheet_name="Orders", index=False)
print("Trade log exported to", excel_filename)

