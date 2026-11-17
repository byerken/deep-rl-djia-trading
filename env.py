"""Trading environment, kept close to the paper (Yang et al., 2020, "An Ensemble Strategy").

One step = one trading day. The agent sees the market after the close of day t and its trades
are executed at the close of day t+1, so it never trades on information it couldn't have had.
"""
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd

N = 30              # stock slots: the 30 DJIA members of each day
V0 = 1_000_000      # starting cash (paper)
FEE = 0.001         # 0.1% of every trade (paper)
MAX_TRADE = 0.01    # a stock can be traded for at most 1% of the portfolio per day (~ paper's 100 shares on $1M)
PUT_DAYS = 21       # 'put' rule: buy 1-month at-the-money SPY puts ...
PUT_COST = 0.01     # ... paying 1% of the premium on top as trading cost
RULES = ['none', 'sell', 'put']   # what to do when turbulence >= threshold


def load_data():
    """Turn the Phase 1 files into (day x slot) arrays that the env can index quickly."""
    prices = pd.read_parquet('data/prices.parquet')
    membership = pd.read_parquet('data/membership.parquet')
    asset = pd.read_parquet('data/asset_features.parquet')
    market = pd.read_parquet('data/market_features.parquet')
    rf = pd.read_parquet('data/rf.parquet').set_index('date').rf
    dates = market.index
    rows = np.arange(len(dates))[:, None]

    # slots: a stock keeps its slot while it's in the DJIA; a newcomer takes the slot of the stock it replaced
    snaps = membership.groupby('date').permno.apply(list)
    slots, current = [], None
    for members in snaps:
        new = sorted(set(members) - set(current or []))
        current = new if current is None else [p if p in members else new.pop(0) for p in current]
        slots.append(current)
    slot = np.array(slots)[snaps.index.searchsorted(dates, side='right') - 1]     # day x 30 permnos

    ret = prices.pivot(index='date', columns='permno', values='ret').reindex(dates)
    sidx = ret.columns.get_indexer(slot.ravel()).reshape(slot.shape)            # slot -> column
    assert (sidx >= 0).all()

    # total-return price (dividends and delisting returns included); stays flat after a delisting
    tr = (1 + ret.fillna(0)).cumprod().values
    raw_close = prices.pivot(index='date', columns='permno', values='close').reindex(dates)

    def per_slot(x):   # (day x stock) -> (day x slot)
        return x.reindex(index=dates, columns=ret.columns).values[rows, sidx]

    close = per_slot(asset['close'].unstack())
    feats = np.stack([per_slot(asset['macd'].unstack()) / close * 100,     # MACD in % of the price
                      (per_slot(asset['rsi'].unstack()) - 50) / 50,         # RSI  -> [-1, 1]
                      per_slot(asset['cci'].unstack()) / 100,
                      per_slot(asset['adx'].unstack()) / 50], axis=1)

    # price-weighted DJIA, rebuilt from its members (total return): the paper's benchmark
    w = raw_close.shift().values[rows, sidx]
    r = ret.values[rows, sidx]
    w = np.where(np.isnan(r), 0, np.nan_to_num(w))
    djia = pd.Series((w * np.nan_to_num(r)).sum(axis=1) / w.sum(axis=1).clip(min=1e-9), index=dates)

    spy = prices[prices.ticker == 'SPY'].set_index('date').close
    return SimpleNamespace(
        dates=dates,
        slot=slot,                                            # permno in each slot
        px=tr[rows, sidx],                                    # price of the stock in each slot
        px_carry=np.vstack([tr[:1, sidx[0]], tr[rows[1:], sidx[:-1]]]),   # today's price of yesterday's slot stock
        stock_ret=np.nan_to_num(r),                           # daily return of each slot stock
        tradable=~np.isnan(close),                            # in the DJIA and trading that day
        feats=np.nan_to_num(feats).reshape(len(dates), -1),   # MACD x30, RSI x30, CCI x30, ADX x30
        ret_all=ret.values, sidx=sidx,                        # every stock's returns (for the min-variance baseline)
        turbulence=market.turbulence.values,
        vix=market.vix.ffill().values,
        rf=rf.reindex(dates).ffill().values,                  # daily risk-free return
        spy=spy.reindex(dates).ffill().values,
        djia=djia,
    )


def turbulence_threshold(data, val_start):
    """The paper's rule, using only data before the validation window:
    99th percentile of past turbulence, or the 90th if the last quarter was already turbulent."""
    past = data.turbulence[:data.dates.searchsorted(pd.Timestamp(val_start))]
    past = past[~np.isnan(past)]
    q90, q99 = np.quantile(past, [0.90, 0.99])
    return q90 if past[-63:].mean() > q90 else q99


def bs_put(S, K, T, r, sigma):
    """Black-Scholes price of a European put (T in years)."""
    if T <= 0:
        return max(K - S, 0.0)
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * cdf(-d2) - S * cdf(-d1)


class TradingEnv(gym.Env):
    """Trades the DJIA members over the days in [start, end).

    State (181, same layout as the paper, scaled to roughly [-1, 1]):
        cash share, 30 daily returns (%), 30 portfolio weights (x30), 30 MACD, 30 RSI, 30 CCI, 30 ADX
    Action: 30 numbers in [-1, 1]; +1 = buy 1% of the portfolio of that stock, -1 = sell 1%.
    Reward: the day's portfolio return in %.
    rule: what happens when turbulence >= threshold. 'none': nothing; 'sell': sell everything and
    don't buy (the paper); 'put': keep the stocks and buy 1-month SPY puts as insurance.
    """

    def __init__(self, data, start, end, rule='none', threshold=np.inf):
        self.d, self.rule, self.threshold = data, rule, threshold
        self.t_start = max(data.dates.searchsorted(pd.Timestamp(start)) - 1, 0)   # state day before the first trading day
        self.t_end = data.dates.searchsorted(pd.Timestamp(end)) - 1               # last trading day before `end`
        self.observation_space = gym.spaces.Box(-10, 10, (1 + 6 * N,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (N,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = (options or {}).get('portfolio')     # carry the portfolio over from the previous quarter
        self.t = self.t_start
        self.cash = p['cash'] if p else V0
        self.h = p['h'].copy() if p else np.zeros(N)       # units held of each slot's total-return price
        self.put = p['put'] if p else None                 # {'q': units of SPY, 'K': strike, 'expiry': day}
        self.value = self._value()
        return self._obs(), {}

    def portfolio(self):
        return {'cash': self.cash, 'h': self.h.copy(), 'put': self.put}

    def step(self, action):
        d = self.d
        turbulent = d.turbulence[self.t] >= self.threshold    # known at the close of the state day
        t = self.t = self.t + 1

        # 1. a stock that left the DJIA or delisted is sold (its slot then belongs to the newcomer)
        gone = (d.slot[t] != d.slot[t - 1]) | ~d.tradable[t]
        self.cash += self.h[gone] @ d.px_carry[t, gone] * (1 - FEE)
        self.h[gone] = 0

        # 2. an expired put pays out
        if self.put and t >= self.put['expiry']:
            self.cash += self.put['q'] * max(self.put['K'] - d.spy[t], 0)
            self.put = None

        # 3. crash rule, then the agent's trades
        if turbulent and self.rule == 'sell':
            self.cash += self.h @ d.px[t] * (1 - FEE)
            self.h[:] = 0
        else:
            if turbulent and self.rule == 'put' and self.put is None:
                self._buy_put(t)
            self._trade(np.asarray(action), t)

        new_value = self._value()
        reward = 100 * (new_value / self.value - 1)
        self.value = new_value
        return self._obs(), reward, t >= self.t_end, False, {'value': new_value, 'turbulent': turbulent}

    def _trade(self, action, t):
        """Sell first, then buy (as in the paper). If cash runs out, all buys are scaled down."""
        price = self.d.px[t]
        amount = np.where(self.d.tradable[t], action, 0) * MAX_TRADE * self._value()   # $ per stock
        sell = np.minimum(-amount.clip(max=0), self.h * price)
        self.h -= sell / price
        self.cash += sell.sum() * (1 - FEE)
        buy = amount.clip(min=0)
        budget = max(self.cash, 0) / (1 + FEE)
        if buy.sum() > budget:
            buy *= budget / buy.sum()
        self.h += buy / price
        self.cash -= buy.sum() * (1 + FEE)

    def _buy_put(self, t):
        """Insure the stock positions with at-the-money SPY puts (priced with VIX and the T-bill rate)."""
        S, stocks = self.d.spy[t], self.h @ self.d.px[t]
        if stocks <= 0:
            return
        q = stocks / S
        self.cash -= q * bs_put(S, S, PUT_DAYS / 252, self.d.rf[t] * 252, self.d.vix[t] / 100) * (1 + PUT_COST)
        self.put = {'q': q, 'K': S, 'expiry': t + PUT_DAYS}

    def _value(self):
        d, t = self.d, self.t
        value = self.cash + self.h @ d.px[t]
        if self.put:
            T = (self.put['expiry'] - t) / 252
            value += self.put['q'] * bs_put(d.spy[t], self.put['K'], T, d.rf[t] * 252, d.vix[t] / 100)
        return value

    def _obs(self):
        d, t = self.d, self.t
        weights = self.h * d.px[t] / self.value
        obs = np.concatenate([[self.cash / self.value], d.stock_ret[t] * 100, weights * N, d.feats[t]])
        return np.clip(obs, -10, 10).astype(np.float32)


def run_episode(model, env, portfolio=None):
    """Let a trained agent trade through env.
    Returns a DataFrame (daily return, turbulent flag) and the final portfolio, to carry into the next quarter."""
    obs, _ = env.reset(options={'portfolio': portfolio})
    values, turbulent, done = [env.value], [False], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        values.append(info['value'])
        turbulent.append(info['turbulent'])
    days = env.d.dates[env.t_start: env.t_end + 1]
    out = pd.DataFrame({'ret': pd.Series(values).pct_change().values, 'turbulent': turbulent}, index=days)
    return out.iloc[1:], env.portfolio()
