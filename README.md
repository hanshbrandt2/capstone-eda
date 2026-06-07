# Intraday regime classification in WTI crude oil futures

Capstone initial report and exploratory data analysis (Module 20.1). The
deliverable is the notebook, with all outputs embedded so it renders on
GitHub without being run.

Notebook: [eda.ipynb](eda.ipynb)

## The question

Can the next 60 minutes of WTI crude be sorted into a regime (trending up,
trending down, or mean-reverting) from what the market has already done?
The label has three states (UP, DOWN, MR); ambiguous windows are dropped as
NO_TRADE.

A regime is defined in two dimensions, not one. A window is UP or DOWN only
if the forward vol-adjusted return is large and the forward path is
efficient: a clean move, not a round trip. MR needs a small move and a
choppy path. The thresholds are quantiles of the training window, not fixed
numbers, so the label adapts to the volatility of its own era.

The premise under test is persistence. If the recent path looked like a
clean trend, does the next hour tend to as well? The headline feature, TPS
(Trend Persistence Score), measures exactly that, pointed backward. The
project asks whether backward persistence predicts the forward regime, and
reports the answer honestly whichever way it falls.

## The data

| | |
|---|---|
| Source | CME, 1-minute OHLCV bars on the front-month WTI contract |
| Window | 2010-06-06 to 2025-12-12 |
| Sessions | 3,942 |
| Bars | 5,280,818 |
| Columns | 73 after feature engineering |
| Shards | 8 parquet files, each under 100 MB, committed to the repo |

The matrix is included here, sharded by year-range so each file stays under
GitHub's 100 MB per-file limit. The notebook loads them with one
glob-and-concatenate, so the whole analysis runs end to end from a clean
clone. A full run takes 15 to 20 minutes; the 173-fold walk-forward
dominates that. Every figure is also written to images/, so the visuals can
be read without running anything.

Feature families, and why each is here:

- Price and momentum. Multi-horizon log returns, realised volatility, ATR,
  vol-adjusted momentum. The immediate state of the move.
- TPS. The backward Trend Persistence Score: a vol-adjusted return times a
  path efficiency, at five horizons. The persistence premise made
  measurable.
- Term structure. Slope and curvature of the futures curve. Contango versus
  backwardation is the market's read on near-term supply.
- Open interest. Positioning, not flow. Rising price on rising OI is fresh
  conviction; on falling OI it is short-covering. Published once a day, so
  it is carried at the prior session's value.
- Volume. Liquidity and abnormal activity. A signal that is not confirmed
  in volume is not tradeable.
- Vol surface. ATM implied volatility, the vol term structure, and skew,
  from the options grid. The options market's price of risk. Daily, carried
  at the prior session's settle.
- Intraday session. Time-of-day flags. Liquidity across the 23-hour day
  varies about 70 times peak to trough, and a signal in thin Asian hours is
  not the same as one at the US settle.

## Why this is hard

Front-month WTI is one of the most liquid and most heavily arbitraged
contracts in the world. Something as simple as one-hour persistence has
been traded against for decades. The honest prior is that little is left to
find at this horizon, and the analysis is built to report that if it is
true rather than to manufacture an edge.

The data carries the usual frictions of a real futures market, and the
notebook treats each one rather than smoothing it away:

- Rolls. A monthly contract has to be rolled to the next. The price can
  jump at the roll, and any feature window that straddles one is
  contaminated. Rows within two sessions of a roll, and any whose forward
  window crosses a roll, are dropped, about 23 percent of bars.
- Settlement. The most liquid two minutes of the day is the NYMEX settle.
  The daily features (open interest, term structure, implied vol) are
  end-of-session quantities, carried at the prior session's settle, the
  most recent value knowable intraday.
- Microstructure. Prices move on a one-cent tick. Overnight, many minutes
  have no trade at all, so a one-minute return is often exactly zero. The
  vol-adjusted, efficiency-based label is built to survive this, where a
  raw-return threshold would mostly be detecting volatility.
- Non-stationarity. Sixteen years hold several distinct regimes. A single
  in-sample fit would flatter itself. The baseline is walk-forward,
  retrained every month, so nothing from the future touches a fold.

## What the analysis found

Three macro regimes leave coherent signatures across price, realised vol
and implied vol at the same time: the 2014-16 OPEC share war, the
March-April 2020 COVID collapse including the negative-WTI settle, and the
February-March 2022 invasion of Ukraine. The fifteen largest daily moves
are each explained by a named catalyst. Outliers here are events, not data
errors, and are kept.

On the persistence question the answer is negative. Backward TPS, built to
spec and correctly aligned, ranks the forward regime at about 0.50 AUC at
every horizon tested (60, 120, 240 minutes), and momentum autocorrelation
is about zero. No feature in the set ranks the regime more than about 0.02
AUC above chance.

That is the expected shape for a series like this. Returns carry no serial
structure while volatility clusters, so the regime label, which lives on
the direction axis, has little to predict from. The result also survives
the 60-minute window overlap: on non-overlapping windows the ranking stays
at chance, so it is not an artefact of the effective sample being far
smaller than the row count.

The baseline confirms it. Walk-forward multinomial logistic regression with
L2, 173 folds (12-month train, 1-month test, monthly step), labels assigned
per training fold, three classes, pooled out-of-sample:

| Metric (pooled out-of-sample) | Value |
|---|---|
| Accuracy | 0.440 |
| Macro F1 | 0.242 |
| Macro ROC-AUC (one-vs-rest) | 0.527 |
| Majority-class accuracy | 0.447 |

The model barely clears the floor. Accuracy sits at or below
always-guessing the majority class; macro-F1 edges the trivial baselines
only by spreading its predictions across the three classes; macro ROC-AUC
is a few points above a coin flip. For a market this efficient that is the
expected result.

That is the answer to the research question, and it is the point of a
baseline: this logistic regression is the floor. The next phase tests
whether richer features the current matrix cannot carry (intraday
open-interest and skew dynamics, calendar-spread divergence) can clear it.

## Evaluation metric

Macro-averaged F1 is the headline: compute F1 for each class, then take the
unweighted mean of the three. Equal weight per class, so a model cannot
score well by riding the majority the way it can on accuracy. Per-class
ROC-AUC sits alongside as the threshold-free ranking measure, the honest
read when the argmax cut is hard. Accuracy is reported but is not the
headline on an imbalanced multi-class target.
