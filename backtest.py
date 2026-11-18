"""Backtest (as in the paper): trade the test quarters one after another with the trained agents,
carrying the portfolio over, for every strategy (ensemble or a single agent) and crash rule.
Also builds the benchmarks: the DJIA and a minimum-variance portfolio (paper), 1/N and a random agent,
and a sweep of crash rules, turbulence thresholds and fees for the ensemble (seeds 0-4, baseline).

Run after train.py (about 20 minutes on 10 cores):  python3 backtest.py
Saves results/backtest.parquet, results/baselines.parquet and results/sweep.parquet
"""
import os

for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[var] = '1'

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from agents import AGENTS, load_agent
from env import EXTRA_RULES, FEE, RULES, TradingEnv, load_data, run_episode, turbulence_threshold

STRATEGIES = ['ensemble'] + AGENTS
CONFIGS = {rule: (rule, None, FEE) for rule in RULES}       # name: (crash rule, threshold quantile, fee)
SWEEP = ({rule: (rule, None, FEE) for rule in EXTRA_RULES}                          # paper's threshold
         | {f'{rule}@q{q}': (rule, q, FEE) for rule in ['sell', 'put', 'half'] for q in [0.9, 0.95, 0.975, 0.99, 0.995]}
         | {f'none@fee{fee}': ('none', None, fee) for fee in [0, 0.0005, 0.002]})


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
    """Metrics for each group (averaged over the seeds) and for the benchmarks."""
    rows = {}
    for name, g in results.groupby(by):
        per_seed = pd.DataFrame([metrics(x.set_index('date').ret.loc[start:end], baselines.rf)
                                 for _, x in g.groupby('seed')])
        rows[name] = per_seed.mean().to_dict() | {'Sharpe std over seeds': per_seed['Sharpe (rf = 0)'].std()}
    for b in baselines.columns.drop('rf'):
        rows[b] = metrics(baselines[b].loc[start:end], baselines.rf).to_dict()
    return pd.DataFrame(rows).T


def pretty(table):
    """Show returns, volatility and drawdowns in %, Sharpe ratios with 2 decimals."""
    sharpe_cols = [c for c in table if 'Sharpe' in c]
    pct_cols = [c for c in table if c not in sharpe_cols]
    return table.style.format('{:.1%}', subset=pct_cols, na_rep='').format('{:.2f}', subset=sharpe_cols, na_rep='')


class RandomAgent:
    """Benchmark: trades at random."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, 30), None


def backtest(data, folds, validation, variant, seed, strategies=STRATEGIES, configs=CONFIGS):
    """Daily returns of every strategy x crash rule (config) for one trained variant and seed,
    over the folds that run was trained on (seeds 5-19 start in 2016 with $1M in cash)."""
    val = validation[(validation.variant == variant) & (validation.seed == seed)]
    portfolio = {(s, c): None for s in strategies for c in configs}
    out = []
    for fold, f in folds.loc[sorted(val.fold.unique())].iterrows():
        best = val[val.fold == fold].set_index('agent').val_sharpe.idxmax()   # ensemble: best validation Sharpe
        models = {a: load_agent(f'models/{variant}/seed{seed}/fold{fold:03d}_{a}.zip') for a in AGENTS}
        models['random'] = RandomAgent([seed, fold])
        for s in strategies:
            agent = best if s == 'ensemble' else s
            for name, (rule, q, fee) in configs.items():
                env = TradingEnv(data, f.test_start, f.test_end, rule, turbulence_threshold(data, f.val_start, q), fee)
                res, portfolio[s, name] = run_episode(models[agent], env, portfolio[s, name])
                out.append(res.assign(variant=variant, seed=seed, strategy=s, rule=name, fold=fold, agent=agent))
    return pd.concat(out).rename_axis('date').reset_index()


def min_variance(data, folds, equal=False):
    """Long-only minimum-variance portfolio of each quarter's DJIA members, from the past year of returns.
    equal=True: 1/N instead (equal weights, rebalanced daily)."""
    out = []
    for f in folds.itertuples():
        t0, t1 = data.dates.searchsorted([f.test_start, f.test_end])
        past = pd.DataFrame(data.ret_all[max(t0 - 252, 0):t0, data.sidx[t0]])
        cov = past.cov().fillna(0).values * 1e4                       # in %^2 so the optimizer is well scaled
        bounds = [(0, 1) if n >= 63 else (0, 0) for n in past.count()]   # skip stocks with < 3 months of history
        w = np.ones(30) / 30 if equal else minimize(lambda w: w @ cov @ w, np.ones(30) / 30, bounds=bounds,
                                                     constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}).x
        r = np.nan_to_num(data.ret_all[t0:t1, data.sidx[t0]]) @ w
        out.append(pd.Series(r, index=data.dates[t0:t1]))
    return pd.concat(out)


data = None


def job(variant, seed, kind):
    """One backtest run (kind: 'main', 'sweep' or 'random'), in its own process."""
    global data
    data = data if data is not None else load_data()
    folds, validation = load_folds(), pd.read_csv('results/validation.csv')
    if kind == 'main':
        return backtest(data, folds, validation, variant, seed)
    if kind == 'sweep':
        return backtest(data, folds, validation, variant, seed, ['ensemble'], SWEEP)
    return backtest(data, folds, validation, variant, seed, ['random'], {'none': CONFIGS['none']})


if __name__ == '__main__':
    folds, validation = load_folds(), pd.read_csv('results/validation.csv')
    runs = list(validation.groupby(['variant', 'seed']).groups)
    jobs = ([(v, s, 'main') for v, s in runs] + [('baseline', s, 'sweep') for s in range(5)]
            + [('baseline', s, 'random') for s in range(5)])
    with ProcessPoolExecutor(10) as pool:
        out = list(pool.map(job, *zip(*jobs)))
    main, sweep, rand = out[:len(runs)], out[len(runs):-5], out[-5:]
    pd.concat(main).to_parquet('results/backtest.parquet', compression='zstd')
    pd.concat(sweep).to_parquet('results/sweep.parquet', compression='zstd')

    data = load_data()
    start = folds.test_start.iloc[0]
    baselines = pd.DataFrame({'DJIA': data.djia, 'Min-variance': min_variance(data, folds),
                              '1/N': min_variance(data, folds, equal=True),
                              'Random agent': pd.concat(rand).groupby('date').ret.mean(),
                              'rf': pd.Series(data.rf, index=data.dates)})
    baselines.loc[start:].to_parquet('results/baselines.parquet')
    print('saved results/backtest.parquet, results/sweep.parquet and results/baselines.parquet')
