import os
import time
import numpy as np
import pandas as pd
from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.common.vec_env import DummyVecEnv

from bitcoin.config import (
    TRAINED_MODEL_DIR, RESULTS_DIR,
    REBALANCE_WINDOW, VALIDATION_WINDOW,
    TRAIN_END_DATE, VAL_END_DATE, END_DATE,
)
from bitcoin.data_loader import data_split
from bitcoin.env_train import BitcoinEnvTrain
from bitcoin.env_validation import BitcoinEnvValidation
from bitcoin.env_trade import BitcoinEnvTrade

os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_A2C(env_train, model_name, timesteps=30000):
    t0 = time.time()
    model = A2C("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print(f"A2C trained in {(time.time()-t0)/60:.2f} min")
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    t0 = time.time()
    model = PPO2("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print(f"PPO trained in {(time.time()-t0)/60:.2f} min")
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    t0 = time.time()
    model = DDPG("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print(f"DDPG trained in {(time.time()-t0)/60:.2f} min")
    return model


def validate(model, val_data, val_env, val_obs):
    for _ in range(len(val_data)):
        action, _ = model.predict(val_obs)
        val_obs, _, done, _ = val_env.step(action)
        if done:
            break


def sharpe_from_csv(iteration):
    path = f"{RESULTS_DIR}/account_value_validation_{iteration}.csv"
    df = pd.read_csv(path, index_col=0)
    df.columns = ["account_value"]
    ret = df["account_value"].pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return (365 ** 0.5) * ret.mean() / ret.std()


def trade(df, model, model_name, last_state, iteration,
          unique_dates, turbulence_threshold, initial):
    start_date = unique_dates[max(iteration - REBALANCE_WINDOW, 0)]
    end_date = unique_dates[min(iteration, len(unique_dates) - 1)]
    trade_data = data_split(df, start=str(start_date.date()), end=str(end_date.date()))

    env = DummyVecEnv([lambda: BitcoinEnvTrade(
        trade_data,
        turbulence_threshold=turbulence_threshold,
        initial=initial,
        previous_state=last_state,
        model_name=model_name,
        iteration=iteration,
    )])
    obs = env.reset()
    for i in range(len(trade_data)):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if i == len(trade_data) - 2:
            last_state = env.envs[0].render()
        if done:
            break
    return last_state


def run_ensemble_strategy(df: pd.DataFrame) -> None:
    print("===== Bitcoin Ensemble Strategy =====")
    df = df.copy()
    df.index = pd.to_datetime(df["date"] if "date" in df.columns else df.index)

    unique_dates = pd.Series(df.index.unique()).sort_values().reset_index(drop=True)
    n = len(unique_dates)

    # turbulence threshold based on price volatility (rolling std of returns)
    if "turbulence" not in df.columns:
        df["turbulence"] = 0.0  # no turbulence guard if not computed

    turbulence_threshold = np.quantile(
        df["turbulence"].values, 0.90
    ) if df["turbulence"].max() > 0 else np.inf

    last_state = []
    initial = True
    t_total = time.time()

    start = REBALANCE_WINDOW + VALIDATION_WINDOW
    for i in range(start, n, REBALANCE_WINDOW):
        win_start = unique_dates[i - REBALANCE_WINDOW - VALIDATION_WINDOW]
        val_start  = unique_dates[i - REBALANCE_WINDOW]
        trade_end  = unique_dates[min(i, n - 1)]

        print(f"\n--- Window {i}: train→{win_start.date()} val→{val_start.date()} trade→{trade_end.date()} ---")

        train_data = data_split(df, start=str(df.index.min().date()), end=str(win_start.date()))
        val_data   = data_split(df, start=str(win_start.date()), end=str(val_start.date()))

        env_train = DummyVecEnv([lambda: BitcoinEnvTrain(train_data)])
        env_val   = DummyVecEnv([lambda: BitcoinEnvValidation(
            val_data, turbulence_threshold=turbulence_threshold, iteration=i)])
        obs_val = env_val.reset()

        model_a2c  = train_A2C(env_train,  f"BTC_A2C_{i}",  timesteps=30000)
        validate(model_a2c,  val_data, env_val, obs_val)
        sharpe_a2c = sharpe_from_csv(i)
        print(f"  A2C  Sharpe: {sharpe_a2c:.4f}")

        env_val.reset()
        model_ppo  = train_PPO(env_train,  f"BTC_PPO_{i}",  timesteps=50000)
        validate(model_ppo,  val_data, env_val, obs_val)
        sharpe_ppo = sharpe_from_csv(i)
        print(f"  PPO  Sharpe: {sharpe_ppo:.4f}")

        env_val.reset()
        model_ddpg = train_DDPG(env_train, f"BTC_DDPG_{i}", timesteps=10000)
        validate(model_ddpg, val_data, env_val, obs_val)
        sharpe_ddpg = sharpe_from_csv(i)
        print(f"  DDPG Sharpe: {sharpe_ddpg:.4f}")

        best_sharpe = max(sharpe_a2c, sharpe_ppo, sharpe_ddpg)
        if best_sharpe == sharpe_ppo:
            model_best, name_best = model_ppo, "PPO"
        elif best_sharpe == sharpe_a2c:
            model_best, name_best = model_a2c, "A2C"
        else:
            model_best, name_best = model_ddpg, "DDPG"
        print(f"  Selected: {name_best}")

        last_state = trade(
            df=df, model=model_best, model_name="ensemble",
            last_state=last_state, iteration=i,
            unique_dates=unique_dates,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )
        initial = False

    print(f"\nTotal time: {(time.time()-t_total)/60:.2f} min")
