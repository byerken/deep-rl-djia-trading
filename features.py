"""Phase 1 feature builder, kept close to the paper. Every value at date t uses data up to t only
(checked by test_lookahead.py)."""
import numpy as np
import pandas as pd


def wilder(x, n):
    """Wilder's smoothing (used by RSI and ADX): EMA with alpha = 1/n."""
    return x.ewm(alpha=1/n, adjust=False).mean()


def build_features(prices, membership, vix):
    """prices: CRSP daily rows, membership: DJIA snapshots (date, permno), vix: (date, vix).
    Returns (asset, market): asset has one row per (date, DJIA member), market one row per date."""

    def wide(col):   # dates x permnos
        return prices.pivot(index='date', columns='permno', values=col)

    # split-adjusted prices (divide by CRSP's split factor), total log returns from CRSP
    f = wide('cumfacpr')
    close = wide('close') / f
    high = (wide('high') / f).fillna(close)
    low = (wide('low') / f).fillna(close)
    lr = np.log1p(wide('ret'))
    dates, assets = close.index, close.columns
    ticker = prices.groupby('permno').ticker.last()
    spy, tlt = ticker[ticker == 'SPY'].index[0], ticker[ticker == 'TLT'].index[0]

    # asset mask: DJIA member on t (latest snapshot on or before t) and has a price on t
    member = (membership.assign(x=True)
              .pivot_table(index='date', columns='permno', values='x', aggfunc='any', fill_value=False)
              .reindex(dates, method='ffill').reindex(columns=assets)
              .fillna(False).astype(bool))
    mask = member & close.notna()

    # ---------- per-stock features: the paper's 4 indicators ----------
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()

    delta = close.diff()
    gain = wilder(delta.clip(lower=0), 14)
    loss = wilder(-delta.clip(upper=0), 14)
    rsi = 100 - 100 / (1 + gain / loss)

    tp = (high + low + close) / 3
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad)

    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift()
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = wilder(tr, 14)
    plus_di = 100 * wilder(plus_dm, 14) / atr
    minus_di = 100 * wilder(minus_dm, 14) / atr
    adx = wilder(100 * (plus_di - minus_di).abs() / (plus_di + minus_di), 14)

    active = mask.stack()
    active = active[active].index
    feats = {'close': close, 'macd': macd, 'rsi': rsi, 'cci': cci, 'adx': adx}
    asset = pd.DataFrame({name: x.stack().reindex(active) for name, x in feats.items()})
    asset = asset.replace([np.inf, -np.inf], np.nan)
    asset.insert(0, 'ticker', asset.index.get_level_values('permno').map(ticker))

    # ---------- market features ----------
    # turbulence (as in the paper): how unusual today's returns of the 30 members are,
    # compared with their mean and covariance over the past 252 days
    turbulence = pd.Series(np.nan, index=dates)
    for i in range(252, len(dates)):
        cols = mask.columns[mask.iloc[i]]
        past = lr[cols].iloc[i - 252:i].dropna(axis=1)   # skip stocks without a full year of history
        d = lr[past.columns].iloc[i] - past.mean()
        turbulence.iloc[i] = d @ np.linalg.pinv(past.cov().values) @ d

    market = pd.DataFrame({
        'vix': vix.set_index('date').vix.reindex(dates).ffill(),
        'turbulence': turbulence,
        'dispersion': lr.where(mask).std(axis=1),   # spread of today's DJIA stock returns
        'stock_bond_corr': lr[spy].rolling(63).corr(lr[tlt]),
        'mkt_ret_21': lr[spy].rolling(21).sum(),
    })
    return asset, market
