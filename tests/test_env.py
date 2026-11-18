"""Sanity checks for the trading environment's accounting."""
import numpy as np

from env import N, V0, TradingEnv, bs_put, load_data

data = load_data()


def run(env, policy):
    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(policy())
    return env, info


def test_doing_nothing_keeps_the_cash():
    env, _ = run(TradingEnv(data, '2008-01-01', '2009-01-01'), lambda: np.zeros(N))
    assert env.value == V0


def test_sell_rule_never_holds_stocks_when_turbulent():
    env, _ = run(TradingEnv(data, '2008-01-01', '2009-01-01', 'sell', threshold=-np.inf), lambda: np.ones(N))
    assert env.h.sum() == 0 and env.value == V0


def test_random_trading_over_all_years_stays_consistent():
    rng = np.random.default_rng(0)
    for rule in ['none', 'sell', 'put']:
        env, info = run(TradingEnv(data, '2000-01-01', '2026-01-01', rule, threshold=200), lambda: rng.uniform(-1, 1, N))
        assert np.isfinite(info['value']) and info['value'] > 0
        assert (env.h >= -1e-9).all()


def test_put_protects_in_a_crash():
    # fully invested on 2008-09-15 (Lehman); a 1-month ATM put bought then must have paid off a month later
    env = TradingEnv(data, '2008-09-15', '2008-10-16', 'put', threshold=-np.inf)
    env.reset()
    env.h = np.full(N, 0.95 * V0 / N) / data.px[env.t]
    env.cash = 0.05 * V0
    env.value = env._value()
    env.step(np.zeros(N))
    premium_paid = 0.05 * V0 - env.cash
    for _ in range(PUT_DAYS := 21):
        env.step(np.zeros(N))
    assert premium_paid > 0 and env.put is not None and env.cash > 0.05 * V0   # first put expired in the money


def test_black_scholes_put():
    assert abs(bs_put(100, 100, 1, 0.05, 0.2) - 5.5735) < 1e-3
    assert bs_put(100, 110, 0, 0.05, 0.2) == 10


def test_caps_of_the_gentler_rules():
    # 'half' keeps at most 50% in stocks while turbulent; 'volscale' at most the volatility cap; all rules stay consistent
    rng = np.random.default_rng(0)
    for rule in ['half', 'volscale', 'put95', 'put3m']:
        env = TradingEnv(data, '2008-06-01', '2009-06-01', rule, threshold=-np.inf)
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(rng.uniform(0, 1, N))     # the agent wants to buy every day
            stocks = env.h @ data.px[env.t] / info['value']
            if rule == 'half':
                assert stocks <= 0.5 + 1e-3     # + a hair: the fee on the sale shrinks the portfolio
            if rule == 'volscale':
                assert stocks <= data.vol_cap[env.t - 1] + 1e-3
            assert np.isfinite(info['value']) and info['value'] > 0 and (env.h >= -1e-9).all()
