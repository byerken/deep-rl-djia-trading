"""Learning curves: are the paper's training steps enough?

For a few folds, train each agent for 3x the paper's steps and, every 10% of the way, measure its Sharpe ratio
on the training window (in-sample) and on the validation quarter (out-of-sample, rule 'none').

Run from the terminal (about 30-40 minutes on 10 cores):  python3 curves.py
Saves results/curves.csv
"""
import os

for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[var] = '1'

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from agents import AGENTS, TIMESTEPS, train_agent
from backtest import load_folds, sharpe
from env import TradingEnv, load_data, run_episode

FOLDS = [20, 40, 60, 80, 100]
SEEDS = [0, 1, 2]
CHUNKS, MULT = 15, 3          # 15 checkpoints up to 3x the paper's steps
WORKERS = 10


def curve(fold, seed):
    data, f = load_data(), load_folds().loc[fold]
    rows = []
    for agent in AGENTS:
        for variant in ['baseline', 'fta']:
            train_env = TradingEnv(data, f.train_start, f.val_start)
            model, steps = None, TIMESTEPS[agent] * MULT // CHUNKS
            for i in range(1, CHUNKS + 1):
                if model is None:   # same settings as train.py, then keep learning in chunks
                    model = train_agent(agent, train_env, variant == 'fta', seed, steps)
                else:
                    model.learn(steps, reset_num_timesteps=False)
                train, _ = run_episode(model, TradingEnv(data, f.train_start, f.val_start))
                val, _ = run_episode(model, TradingEnv(data, f.val_start, f.test_start))
                rows.append({'fold': fold, 'seed': seed, 'agent': agent, 'variant': variant, 'steps': i * steps,
                             'paper_steps': TIMESTEPS[agent], 'train_sharpe': sharpe(train.ret),
                             'val_sharpe': sharpe(val.ret)})
    return rows


if __name__ == '__main__':
    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = [pool.submit(curve, fold, seed) for fold in FOLDS for seed in SEEDS]
        rows = [r for job in jobs for r in job.result()]
    pd.DataFrame(rows).to_csv('results/curves.csv', index=False)
    print('saved results/curves.csv')
