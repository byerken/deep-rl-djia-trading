"""Look-ahead test: scramble all data after date t. If features up to t change, they peeked at the future.
Run: pytest test_lookahead.py   (needs data/ from data_collection.ipynb)"""
import numpy as np
import pandas as pd
from features import build_features


def test_no_lookahead():
    prices = pd.read_parquet('data/prices.parquet')
    membership = pd.read_parquet('data/membership.parquet')
    vix = pd.read_parquet('data/vix.parquet')
    prices = prices[prices.date < '2010-01-01']   # smaller sample so the test is quick
    t = pd.Timestamp('2008-06-30')
    rng = np.random.default_rng(0)

    # scramble the future: prices, membership and VIX after t
    cols = ['open', 'high', 'low', 'close', 'ret', 'dollar_vol', 'cumfacpr']
    p2, m2, v2 = prices.copy(), membership.copy(), vix.copy()
    p2.loc[p2.date > t, cols] = rng.permutation(p2.loc[p2.date > t, cols].values)
    m2.loc[m2.date > t, 'permno'] = rng.permutation(m2.loc[m2.date > t, 'permno'].values)
    v2.loc[v2.date > t, 'vix'] = rng.permutation(v2.loc[v2.date > t, 'vix'].values)

    asset1, market1 = build_features(prices, membership, vix)
    asset2, market2 = build_features(p2, m2, v2)

    past1 = asset1[asset1.index.get_level_values('date') <= t]
    past2 = asset2[asset2.index.get_level_values('date') <= t]
    pd.testing.assert_frame_equal(past1, past2)
    pd.testing.assert_frame_equal(market1[:t], market2[:t])
