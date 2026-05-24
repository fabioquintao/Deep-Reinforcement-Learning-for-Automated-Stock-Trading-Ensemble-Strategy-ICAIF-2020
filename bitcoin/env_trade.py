import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from bitcoin.config import INITIAL_BALANCE, TRANSACTION_FEE, HMAX, RESULTS_DIR
from bitcoin.env_train import INDICATORS, OBS_DIM


class BitcoinEnvTrade(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, turbulence_threshold: float,
                 initial: bool, previous_state: list, model_name: str, iteration: int):
        super().__init__()
        self.df = df
        self.day = 0
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.iteration = iteration
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(OBS_DIM,), dtype=np.float32)
        self._reset_state()
        self._seed()

    def _reset_state(self):
        self.data = self.df.iloc[0]
        if self.initial:
            self.balance = INITIAL_BALANCE
            self.btc_held = 0.0
        else:
            self.balance = self.previous_state[0]
            self.btc_held = self.previous_state[1]
        self.terminal = False
        self.asset_memory = [self.balance + self.btc_held * self.data["close"]]
        self.reward = 0.0

    def _get_obs(self):
        obs = [self.balance, self.data["close"], self.btc_held]
        obs += [self.data[ind] for ind in INDICATORS]
        return np.array(obs, dtype=np.float32)

    def _total_asset(self):
        return self.balance + self.btc_held * self.data["close"]

    def step(self, action):
        self.terminal = self.day >= len(self.df) - 1
        if self.terminal:
            import os
            os.makedirs(RESULTS_DIR, exist_ok=True)
            pd.DataFrame(self.asset_memory).to_csv(
                f"{RESULTS_DIR}/account_value_trade_{self.model_name}_{self.iteration}.csv")
            return self._get_obs(), self.reward, True, {}

        price = self.data["close"]
        if "turbulence" in self.df.columns and self.data["turbulence"] > self.turbulence_threshold:
            act = -HMAX
        else:
            act = float(action[0]) * HMAX

        begin_asset = self._total_asset()

        if act < 0:
            sell_amt = min(abs(act), self.btc_held)
            self.balance += price * sell_amt * (1 - TRANSACTION_FEE)
            self.btc_held -= sell_amt
        elif act > 0:
            affordable = self.balance / (price * (1 + TRANSACTION_FEE))
            buy_amt = min(act, affordable)
            self.balance -= price * buy_amt * (1 + TRANSACTION_FEE)
            self.btc_held += buy_amt

        self.day += 1
        self.data = self.df.iloc[self.day]

        self.reward = self._total_asset() - begin_asset
        self.asset_memory.append(self._total_asset())

        return self._get_obs(), self.reward, False, {}

    def reset(self):
        self._reset_state()
        self.day = 0
        self.data = self.df.iloc[0]
        return self._get_obs()

    def render(self, mode="human"):
        return [self.balance, self.btc_held]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
