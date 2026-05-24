import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from bitcoin.config import INITIAL_BALANCE, TRANSACTION_FEE, HMAX

INDICATORS = ["macd", "rsi", "cci", "adx", "boll_ub", "boll_lb", "atr"]
OBS_DIM = 1 + 1 + 1 + len(INDICATORS)   # balance, price, held_btc, indicators


class BitcoinEnvTrain(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.day = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(OBS_DIM,), dtype=np.float32)
        self._reset_state()
        self._seed()

    def _reset_state(self):
        self.balance = INITIAL_BALANCE
        self.btc_held = 0.0
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.asset_memory = [INITIAL_BALANCE]
        self.rewards_memory = []
        self.data = self.df.iloc[0]

    def _get_obs(self):
        obs = [self.balance, self.data["close"], self.btc_held]
        obs += [self.data[ind] for ind in INDICATORS]
        return np.array(obs, dtype=np.float32)

    def _total_asset(self):
        return self.balance + self.btc_held * self.data["close"]

    def step(self, action):
        self.terminal = self.day >= len(self.df) - 1
        if self.terminal:
            return self._get_obs(), self.reward, True, {}

        price = self.data["close"]
        act = float(action[0]) * HMAX   # scale to BTC units

        begin_asset = self._total_asset()

        if act < 0:  # sell
            sell_amt = min(abs(act), self.btc_held)
            self.balance += price * sell_amt * (1 - TRANSACTION_FEE)
            self.btc_held -= sell_amt
            self.cost += price * sell_amt * TRANSACTION_FEE
            if sell_amt > 0:
                self.trades += 1
        elif act > 0:  # buy
            affordable = self.balance / (price * (1 + TRANSACTION_FEE))
            buy_amt = min(act, affordable)
            self.balance -= price * buy_amt * (1 + TRANSACTION_FEE)
            self.btc_held += buy_amt
            self.cost += price * buy_amt * TRANSACTION_FEE
            if buy_amt > 0:
                self.trades += 1

        self.day += 1
        self.data = self.df.iloc[self.day]

        end_asset = self._total_asset()
        self.reward = end_asset - begin_asset
        self.rewards_memory.append(self.reward)
        self.asset_memory.append(end_asset)

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
