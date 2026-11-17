"""Backtest (as in the paper): trade the test quarters one after another with the trained agents,
carrying the portfolio over, for every strategy (ensemble or a single agent) and crash rule.
Also builds the paper's two benchmarks: the DJIA and a minimum-variance portfolio.

Run after train.py (takes a few minutes):  python3 backtest.py
Saves results/backtest.parquet and results/baselines.parquet
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from agents import AGENTS, load_agent
from env import RULES, TradingEnv, load_data, run_episode, turbulence_threshold

STRATEGIES = ['ensemble'] + AGENTS


def load_folds():
    return pd.read_csv('data/folds.csv', parse_dates=['train_start', 'val_start', 'test_start', 'test_end'])


def sharpe(r):
    """Annualised Sharpe ratio of daily returns (rf = 0 as in the paper's code; pass excess returns otherwise)."""
    return 0.0 if r.std() == 0 else r.mean() / r.std() * np.sqrt(252)


def metrics(r, rf):
    """Performance table row for daily returns r (rf: daily risk-free Series)."""
    v = (1 + r).cumprod()
    return pd.Series({
        'Cumulative return': v.iloc[-1] - 1,
        'Annual return': v.iloc[-1] ** (252 / len(r)) - 1,
        'Annual volatility': r.std() * np.sqrt(252),
        'Sharpe (rf = 0)': sharpe(r),
        'Sharpe (excess)': sharpe(r - rf.reindex(r.index)),
        'Max drawdown': (v / v.cummax() - 1).min(),
    })


def summary(results, baselines, start=None, end=None, by='strategy'):
    """Metrics for each group (averaged over the seeds) and for the DJIA and min-variance benchmarks."""
    rows = {}
    for name, g in results.groupby(by):
        per_seed = pd.DataFrame([metrics(x.set_index('date').ret.loc[start:end], baselines.rf)
                                 for _, x in g.groupby('seed')])
        rows[name] = per_seed.mean().to_dict() | {'Sharpe std over seeds': per_seed['Sharpe (rf = 0)'].std()}
    for b in ['DJIA', 'Min-variance']:
        rows[b] = metrics(baselines[b].loc[start:end], baselines.rf).to_dict()
    return pd.DataFrame(rows).T


def pretty(table):
    """Show returns, volatility and drawdowns in %, Sharpe ratios with 2 decimals."""
    sharpe_cols = [c for c in table if 'Sharpe' in c]
    pct_cols = [c for c in table if c not in sharpe_cols]
    return table.style.format('{:.1%}', subset=pct_cols, na_rep='').format('{:.2f}', subset=sharpe_cols, na_rep='')


def backtest(data, folds, validation, variant, seed):
    """Daily returns of every strategy x crash rule for one trained variant and seed."""
    val = validation[(validation.variant == variant) & (validation.seed == seed)]
    portfolio = {(s, rule): None for s in STRATEGIES for rule in RULES}
    out = []
    for fold, f in folds.iterrows():
        best = val[val.fold == fold].set_index('agent').val_sharpe.idxmax()   # ensemble: best validation Sharpe
        models = {a: load_agent(f'models/{variant}/seed{seed}/fold{fold:03d}_{a}.zip') for a in AGENTS}
        threshold = turbulence_threshold(data, f.val_start)
        for s in STRATEGIES:
            agent = best if s == 'ensemble' else s
            for rule in RULES:
                env = TradingEnv(data, f.test_start, f.test_end, rule, threshold)
                res, portfolio[s, rule] = run_episode(models[agent], env, portfolio[s, rule])
                out.append(res.assign(variant=variant, seed=seed, strategy=s, rule=rule, fold=fold, agent=agent))
    return pd.concat(out).rename_axis('date').reset_index()


def min_variance(data, folds):
    """Long-only minimum-variance portfolio of each quarter's DJIA members, from the past year of returns."""
    out = []
    for f in folds.itertuples():
        t0, t1 = data.dates.searchsorted([f.test_start, f.test_end])
        past = pd.DataFrame(data.ret_all[max(t0 - 252, 0):t0, data.sidx[t0]])
        cov = past.cov().fillna(0).values * 1e4                       # in %^2 so the optimizer is well scaled
        bounds = [(0, 1) if n >= 63 else (0, 0) for n in past.count()]   # skip stocks with < 3 months of history
        w = minimize(lambda w: w @ cov @ w, np.ones(30) / 30, bounds=bounds,
                     constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}).x
        r = np.nan_to_num(data.ret_all[t0:t1, data.sidx[t0]]) @ w
        out.append(pd.Series(r, index=data.dates[t0:t1]))
    return pd.concat(out)


if __name__ == '__main__':
    data, folds = load_data(), load_folds()
    validation = pd.read_csv('results/validation.csv')

    # only (variant, seed) runs whose training is complete
    counts = validation.groupby(['variant', 'seed']).size()
    runs = counts[counts == len(folds) * len(AGENTS)].index
    results = []
    for variant, seed in runs:
        print(f'backtesting {variant} seed {seed}', flush=True)
        results.append(backtest(data, folds, validation, variant, seed))
    pd.concat(results).to_parquet('results/backtest.parquet')

    start = folds.test_start.iloc[0]
    baselines = pd.DataFrame({'DJIA': data.djia, 'Min-variance': min_variance(data, folds),
                              'rf': pd.Series(data.rf, index=data.dates)})
    baselines.loc[start:].to_parquet('results/baselines.parquet')
    print('saved results/backtest.parquet and results/baselines.parquet')
