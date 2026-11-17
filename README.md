# Deep RL stock trading on the DJIA: stability with FTA, and options instead of closing out

DSC4107 project. We reproduce the ensemble strategy of
**Yang, Liu, Zhong & Walid (2020), "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"**
on 25 years of point-in-time DJIA data, then test two ideas on top of it:

1. **FTA for stability.** Does a Fuzzy Tiling Activation (Pan, Banman & White, ICLR 2021) in the critic make the agents' learning more stable, so that different random seeds agree more?
2. **Options instead of closing out.** When the turbulence index says a crash may be coming, the paper sells everything. Is it better to keep the stocks and buy SPY put options as insurance?

## Results

Out-of-sample trading on 101 test quarters (Oct 2000 – Dec 2025), averaged over 5 random seeds.

### Phase 2: the paper's ensemble, reproduced (`3_baseline.ipynb`)

| | Annual return | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Ensemble (paper setup) | 5.6% | 15.9% | 0.43 | −41.5% |
| PPO alone | 6.3% | 13.3% | 0.52 | −32.1% |
| DJIA (price-weighted, with dividends) | 8.7% | 18.3% | 0.55 | −51.8% |
| Min-variance | 7.4% | 14.4% | 0.57 | −31.5% |

- **The paper's outperformance doesn't replicate.** Over the paper's own test period (Jan 2016 – May 2020) the ensemble's Sharpe ratio is 0.56, against 1.30 in the paper and 0.61 for the DJIA.
- **What does replicate is lower risk:** smaller drawdowns and lower volatility than the DJIA.
- **The ensemble doesn't beat its best member (PPO).** Choosing an agent from 3 months of validation is noisy.

### Idea 1: FTA doesn't make learning measurably more stable (`4_fta_stability.ipynb`)

- **Seed agreement:** across the 101 test quarters, the 5 seeds disagree about as much with FTA as without. The spread of quarterly returns changes by −10% (DDPG) to +2% (PPO), and no difference is significant (paired Wilcoxon, smallest p = 0.06 for DDPG).
- **Validation:** validation Sharpe ratios vary just as much (all p > 0.8).
- **Performance:** FTA doesn't help either. Ensemble Sharpe is 0.48 against 0.50; DDPG falls from 0.45 to 0.37.
- **DDPG**, the agent closest to the FTA paper's setup, is the only one with a small hint of more stability.

### Idea 2: puts beat closing out, but not doing nothing (`5_options_vs_close.ipynb`)

| Crash rule (ensemble) | Annual return | Sharpe | Max drawdown | 2008 crisis | 2009 rebound |
|---|---|---|---|---|---|
| `none`: ignore turbulence | 8.2% | 0.50 | −56.5% | −47% | +81% |
| `sell`: close out (paper) | 5.6% | 0.43 | −41.5% | −27% | +36% |
| `put`: keep stocks, buy SPY puts | 6.7% | 0.46 | −50.2% | −37% | +53% |

- **Puts vs. selling:** puts beat selling on return and Sharpe ratio, in all 5 seeds of the ensemble and in 27 of the 40 runs.
- **Selling protects best in long crashes.**
- **Neither rule beats ignoring turbulence.** The rule buys insurance after turbulence has spiked, when it's most expensive: at VIX 83 in March 2020, a 1-month put cost 9.5% of the position.

## How to run

```bash
pip install -r requirements.txt
```

| Step | What | Output |
|---|---|---|
| `1_data_collection.ipynb` | Downloads DJIA members, prices, VIX and the T-bill rate from WRDS (CRSP, CBOE, Fama-French) | `data/*.parquet` |
| `2_features.ipynb` | MACD, RSI, CCI, ADX per stock, market features (turbulence, VIX, …), walk-forward folds | `data/asset_features.parquet`, `data/market_features.parquet`, `data/folds.csv` |
| `python3 train.py` | Trains A2C, PPO and DDPG for 101 quarters × 5 seeds × {baseline, FTA} (~2.5 h on 10 cores) | `models/`, `results/validation.csv` |
| `python3 backtest.py` | Trades every test quarter in order, for every strategy and crash rule (~10 min) | `results/backtest.parquet`, `results/baselines.parquet` |
| `3_baseline.ipynb` | Phase 2: the paper's ensemble vs. single agents, DJIA and min-variance | |
| `4_fta_stability.ipynb` | Idea 1: FTA vs. baseline | |
| `5_options_vs_close.ipynb` | Idea 2: crash rules `none` / `sell` / `put` | |
| `python3 -m pytest` | Look-ahead test of the features + accounting tests of the trading env | |

## Code

| File | What's in it |
|---|---|
| `features.py` | Indicators and market features. Every value at date *t* uses data up to *t* only. |
| `env.py` | The trading environment (gymnasium), the turbulence threshold rule and Black-Scholes put pricing |
| `agents.py` | The three agents with the paper's settings, and the FTA layer |
| `train.py` | Walk-forward training and validation, in parallel |
| `backtest.py` | Out-of-sample trading, benchmarks and performance metrics |

## Setup

**Data.** The 30 DJIA members of each day from Oct 1999 to Dec 2025, matched to CRSP by PERMNO, so stocks that later left the index or delisted are included (no survivorship bias). Prices are CRSP total returns, so dividends and delisting returns count.

**Walk-forward (as in the paper).** Every quarter, the agents are trained from scratch on all data since 2000. Each is scored on the next 3 months (validation Sharpe), and the best one trades the quarter after that (test). The portfolio carries over from quarter to quarter. The first test quarter is Oct 2000, which gives 101 test quarters up to Dec 2025.

**Agents.** A2C (10k steps), PPO (50k) and DDPG (20k, Ornstein-Uhlenbeck noise σ = 0.5): the paper's training steps, stable-baselines3 defaults, 64×64 networks.

**Trading environment.** Same 181-number state layout as the paper: cash, 30 prices, 30 holdings, then MACD, RSI, CCI and ADX for each of the 30 stocks. They're expressed scale-free so the networks (and FTA) see numbers around ±1: cash as a share of the portfolio, prices as daily returns, holdings as portfolio weights, and fixed scalings for the indicators. Each trade costs 0.1%. The agent sees the market after the close and trades at the *next* day's close. When a stock leaves the DJIA, it's sold and the newcomer takes its slot.

**Turbulence rule (paper).** The threshold is the 99th percentile of past turbulence, or the 90th if the last quarter was already turbulent. It's computed from training data only.

## Differences from the paper

| | Paper | Here | Why |
|---|---|---|---|
| Stocks | today's 30 DJIA stocks, 2009–2020 | DJIA members of each day, 2000–2025 | no survivorship bias, longer test |
| Prices | adjusted close, no dividends | CRSP total returns | dividends and delistings are real money |
| Timing | trades at the close it just saw | trades at the next close | no look-ahead |
| State | raw $ and share counts | same 181 values, scaled | neural networks (and FTA's fixed tile range) need inputs around ±1 |
| Trade size | ≤ 100 shares per stock per day | ≤ 1% of the portfolio per stock per day | share counts mean nothing across 25 years of splits |
| Reward | $ change | daily % return | comparable across 25 years |
| Library | stable-baselines (TensorFlow 1) | stable-baselines3 (PyTorch) | maintained, same algorithms |

## Limitations

- CRSP ends on 2025-12-31, so the 2026 DJIA change isn't covered.
- The puts are priced with Black-Scholes at VIX volatility, not with real option quotes. VIX is usually a bit above at-the-money implied volatility, so this tends to overstate the cost of insurance.
- The first folds train on only a few months of data (first test quarter Oct 2000).
- 5 seeds per setting: enough to compare stability across 101 quarters, but each single quarter's spread is noisy.
