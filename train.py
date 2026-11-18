"""Train the paper's three agents (A2C, PPO, DDPG) for every fold, seed and variant, and score each on its
validation quarter with the Sharpe ratio (the paper's model selection).

Variants: 'baseline' (the paper), 'fta' (FTA in the critic), 'rolling' (baseline trained on the last 5 years only).
Seeds 0-4 cover all 101 folds; seeds 5-19 only the folds from 2016 on (EXTRA_FROM), to measure seed noise.

Run from the terminal (about 2-3 hours on 10 cores; re-running continues where it stopped):
    python3 train.py
Saves models/<variant>/seed<s>/fold<NNN>_<agent>.zip and results/validation.csv
"""
import os

for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[var] = '1'   # one thread per worker (must be set before numpy/torch load): the workers fill the cores

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from agents import AGENTS, load_agent, train_agent
from backtest import load_folds, sharpe
from env import TradingEnv, load_data, run_episode, turbulence_threshold

VARIANTS = ['baseline', 'fta', 'rolling']
SEEDS = [0, 1, 2, 3, 4]
EXTRA_SEEDS = list(range(5, 20))       # baseline and fta only, folds from EXTRA_FROM on
EXTRA_FROM = '2016-01-01'
WORKERS = 10
OUT = 'results/validation.csv'

folds = load_folds()
data = None


def train_fold(variant, seed, fold):
    global data
    data = data if data is not None else load_data()
    f = folds.loc[fold]
    threshold = turbulence_threshold(data, f.val_start)
    start = max(f.train_start, f.val_start - pd.DateOffset(years=5)) if variant == 'rolling' else f.train_start
    rows = []
    for agent in AGENTS:
        path = f'models/{variant}/seed{seed}/fold{fold:03d}_{agent}.zip'
        if os.path.exists(path):
            model = load_agent(path)
        else:
            model = train_agent(agent, TradingEnv(data, start, f.val_start), variant == 'fta', seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
        val, _ = run_episode(model, TradingEnv(data, f.val_start, f.test_start, 'sell', threshold))
        rows.append({'variant': variant, 'seed': seed, 'fold': fold, 'agent': agent, 'val_sharpe': sharpe(val.ret)})
    return rows


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['variant', 'seed', 'fold'])
    done = set(zip(done.variant, done.seed, done.fold))
    recent = folds.index[folds.test_start >= EXTRA_FROM]
    jobs = ([(v, s, f) for s in EXTRA_SEEDS for v in ['baseline', 'fta'] for f in recent]
            + [(v, s, f) for s in SEEDS for v in VARIANTS for f in folds.index])
    jobs = [job for job in jobs if job not in done]
    print(f'{len(jobs)} folds to train', flush=True)

    start = time.time() 
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(train_fold, *job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            pd.DataFrame(future.result()).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            print(f'{i}/{len(jobs)} done after {(time.time() - start) / 60:.0f} min', flush=True)
