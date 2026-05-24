"""Entry point: train and backtest the Bitcoin ensemble DRL trading bot."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bitcoin.data_loader import load_data, data_split
from bitcoin.models import run_ensemble_strategy
from bitcoin.config import START_DATE, END_DATE, RESULTS_DIR


def plot_results():
    import glob
    files = sorted(glob.glob(f"{RESULTS_DIR}/account_value_trade_ensemble_*.csv"))
    if not files:
        print("No trade result files found.")
        return
    frames = [pd.read_csv(f, index_col=0) for f in files]
    portfolio = pd.concat(frames, ignore_index=True)
    portfolio.columns = ["account_value"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(portfolio["account_value"].values, label="BTC Bot Portfolio")
    plt.title("Bitcoin DRL Trading Bot — Portfolio Value")
    plt.xlabel("Trading Days")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/portfolio_value.png")
    plt.close()
    print(f"Chart saved to {RESULTS_DIR}/portfolio_value.png")

    ret = portfolio["account_value"].pct_change().dropna()
    total_return = (portfolio["account_value"].iloc[-1] / portfolio["account_value"].iloc[0] - 1) * 100
    sharpe = (365 ** 0.5) * ret.mean() / ret.std() if ret.std() > 0 else 0
    print(f"Total return : {total_return:.2f}%")
    print(f"Sharpe ratio : {sharpe:.4f}")


if __name__ == "__main__":
    print(f"Downloading BTC-USD data ({START_DATE} → {END_DATE}) ...")
    df = load_data(start=START_DATE, end=END_DATE)
    # attach a date column for data_split
    df = df.reset_index()   # 'date' becomes a column
    df.index = pd.to_datetime(df["date"])

    print(f"Data loaded: {len(df)} days")
    run_ensemble_strategy(df)
    plot_results()
