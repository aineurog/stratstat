# StratStat Formula Reference — v0.1

> **Phase 1 deliverable.** Every v0.1 metric listed in `stats.md`, with its
> exact formula, citation, and default convention where applicable. This
> document must be reviewed and approved before any metric code is written.

---

## Conventions for Metrics with Competing Definitions

| Metric | Parameter | Options | Default | Citation |
|---|---|---|---|---|
| Sharpe ratio | `ddof` | `0`, `1` | `1` | Sharpe (1994) |
| Sortino ratio | `denominator` | `full_downside`, `downside_only` | `full_downside` | Sortino & Price (1994) |
| Max drawdown | `return_type` | `simple`, `log` | `simple` | Pospisil & Vecer (2011) |
| Drawdown duration | `units` | `periods`, `years` | `periods` | Standard convention |
| VaR | `method` | `historical`, `parametric`, `cornish_fisher` | `historical` | Litterman (1996); Zangari (1996) |
| CVaR / ES | `method` | `historical`, `parametric` | `historical` | Rockafellar & Uryasev (2000) |
| Beta | `variant` | `least_squares` | `least_squares` | Sharpe (1964) |
| Tail ratio | `tail_cutoff` | `float` in `(0, 0.5)` | `0.05` | 95th / 5th percentile split |
| Hill tail index | `tail_fraction` | `float` in `(0, 1)` | `0.10` | Hill (1975); upper 10 % tail |
| PSR | `sharpe_benchmark` | `float` ≥ 0 | `0.0` | Bailey & López de Prado (2012) |
| Kappa-3 | `mar` | float (minimum acceptable return) | `0.0` | Kaplan & Knowles (2004) |
| VaR / CVaR | `confidence` | `float` in `(0, 1)` | `0.95` | Industry standard |

---

## 1. `core.returns` — Descriptive

All tagged `("descriptive", "returns")`, backend `"vectorized"`.

### 1.1 CAGR (Compound Annual Growth Rate)

\[\text{CAGR} = \exp\!\left(\frac{1}{T}\sum_{t=1}^{n}\ln(1+r_t)\right) - 1\]

where \(T\) is the length of the series in years. Equivalent to
\((V_f/V_i)^{1/T} - 1\).

**Citation:** Standard finance (Damodaran, *Investment Valuation*, 3rd ed.,
2012, Ch. 3).

### 1.2 Annualized Volatility

\[\sigma_{\text{ann}} = \sigma \cdot \sqrt{P}\]

where \(\sigma\) is the sample standard deviation (ddof = 1) of period
returns and \(P\) is `periods_per_year`.

**Citation:** CFA Institute, *Quantitative Methods*.

### 1.3 Cumulative Return

\[R_{\text{cum}} = \prod_{t=1}^{n}(1 + r_t) - 1\]

**Citation:** Standard finance textbook.

### 1.4 Arithmetic Mean Return

\[\bar{r} = \frac{1}{n}\sum_{t=1}^{n} r_t\]

**Citation:** Standard statistics.

### 1.5 Geometric Mean Return

\[\bar{r}_g = \left(\prod_{t=1}^{n}(1 + r_t)\right)^{\!1/n} - 1\]

**Citation:** Campbell, Lo & MacKinlay (1997, *The Econometrics of Financial
Markets*, §1.4).

### 1.6 Skewness

\[\gamma_1 = \frac{n}{(n-1)(n-2)}\sum_{t=1}^{n}\!\left(\frac{r_t -
\bar{r}}{\sigma}\right)^{\!3}\]

Adjusted (sample) skewness.

**Citation:** Fisher (1930); bias-corrected divisor `(n-1)(n-2)`.

### 1.7 Excess Kurtosis

\[\gamma_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}
\sum_{t=1}^{n}\!\left(\frac{r_t - \bar{r}}{\sigma}\right)^{\!4}
- \frac{3(n-1)^2}{(n-2)(n-3)}\]

Returns 0 for a normal distribution.

**Citation:** Standard statistics; Fisher kurtosis definition.

### 1.8 Best Period

\[r_{\text{best}} = \max_t r_t\]

At the data's native frequency.

**Citation:** Standard descriptive statistic.

### 1.9 Worst Period

\[r_{\text{worst}} = \min_t r_t\]

At the data's native frequency.

**Citation:** Standard descriptive statistic.

### 1.10 Positive-Period Ratio

\[\text{PPR} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}_{[r_t > 0]}\]

Strictly positive (> 0); zero is non-positive.

**Citation:** Bacon (2008, *Practical Portfolio Performance Measurement and
Attribution*, §3.11).

### 1.11 Autocorrelation (Lag-1)

\[\rho_1 = \frac{\sum_{t=2}^{n}(r_t - \bar{r})(r_{t-1} - \bar{r})}
{\sum_{t=1}^{n}(r_t - \bar{r})^2}\]

**Citation:** Campbell, Lo & MacKinlay (1997, §2.4).

### 1.12 Variance

\[s^2 = \frac{1}{n-1}\sum_{t=1}^{n}(r_t - \bar{r})^2\]

Sample variance (ddof = 1).

**Citation:** Standard statistics.

### 1.13 Return Range

\[R_{\text{range}} = \max_t r_t - \min_t r_t\]

**Citation:** Standard descriptive statistic.

### 1.14 Percentiles

The \(p\)-th percentile at levels \(p \in \{1, 5, 10, 25, 50, 75, 90,
95, 99\}\) of the empirical return distribution, via linear interpolation.

**Citation:** Standard order statistics; Hyndman & Fan (1996).

### 1.15 Coefficient of Variation

\[\text{CV} = \frac{\sigma}{|\bar{r}|}\]

**Citation:** Standard statistics.

### 1.16 Outlier Count & % (IQR Method)

\[r_t \text{ is an outlier if } r_t < Q_1 - 1.5 \times \text{IQR}
\;\text{ or }\; r_t > Q_3 + 1.5 \times \text{IQR}\]

where \(\text{IQR} = Q_3 - Q_1\).

**Citation:** Tukey (1977, *Exploratory Data Analysis*).

---

## 2. `core.returns` — Risk

Tagged `("risk", "returns")`. Backend varies: mostly `"vectorized"`,
drawdown walks are `"sequential"`.

### 2.1 Max Drawdown

\[\text{MDD} = \max_{t}\frac{P_t - \max_{\tau \leq t} P_\tau}
{\max_{\tau \leq t} P_\tau}\]

Peak-to-trough decline on the cumulative return index.

**Citation:** Pospisil & Vecer (2011). Convention parameter: `return_type`
— `"simple"` (default) or `"log"`.

### 2.2 Longest Drawdown Duration

\[T_{\text{DD}}^{\max} = \max_k\!\left(\tau_{\text{end}}^{(k)} -
\tau_{\text{start}}^{(k)}\right)\]

Longest contiguous underwater period in the equity curve.

**Citation:** Standard risk management; van Hemert et al. (2020).

### 2.3 Time to Recovery

For each drawdown episode \(k\), the time from peak to the first subsequent
period where cumulative return reaches or exceeds the previous peak. Report
mean, median, max across episodes.

**Citation:** Standard drawdown analysis.

### 2.4 Average Drawdown

\[\text{ADD} = \frac{1}{K}\sum_{k=1}^{K} D^{(k)}\]

where \(D^{(k)}\) is the peak-to-trough depth of the \(k\)-th episode.

**Citation:** Bacon (2008, §7.2).

### 2.5 Average Drawdown Duration

\[\bar{T}_{\text{DD}} = \frac{1}{K}\sum_{k=1}^{K} T_{\text{DD}}^{(k)}\]

**Citation:** Standard risk metric.

### 2.6 Ulcer Index

\[\text{UI} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}
\left(\frac{P_t - \max_{\tau \leq t} P_\tau}
{\max_{\tau \leq t} P_\tau}\right)^{\!2}}\]

Root-mean-square of percentage drawdowns.

**Citation:** Martin & McCann (1989); Martin (1993).

### 2.7 Downside Deviation

\[\text{DD} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(r_t - \tau,\; 0)^2}\]

With \(\tau = 0\) (MAR), this is semi-deviation.

**Citation:** Sortino & van der Meer (1991); Sortino & Price (1994).

### 2.8 Upside Deviation

\[\text{UD} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\max(r_t - \tau,\; 0)^2}\]

Mirror of downside deviation.

**Citation:** Symmetric counterpart to downside deviation; standard.

### 2.9 VaR (Value at Risk)

**Historical (default):**
\(\text{VaR}_\alpha = -F_r^{-1}(\alpha)\)

**Parametric:**
\(\text{VaR}_\alpha = -(\bar{r} + z_\alpha \cdot \sigma)\)

**Cornish-Fisher:**
Same as parametric with \(z_\alpha\) replaced by the CF expansion:
\(z_{\text{CF}} = z_\alpha + \frac{\gamma_1}{6}(z_\alpha^2 - 1) +
\frac{\gamma_2}{24}(z_\alpha^3 - 3z_\alpha) -
\frac{\gamma_1^2}{36}(2z_\alpha^3 - 5z_\alpha)\)

**Citation:** Litterman (1996); Zangari (1996); Artzner et al. (1999).
Convention parameters: `method` (`"historical"`, `"parametric"`,
`"cornish_fisher"`), `confidence` (default `0.95`).

### 2.10 CVaR / Expected Shortfall

**Historical (default):**
\(\text{CVaR}_\alpha = -\mathbb{E}[r \mid r \leq -\text{VaR}_\alpha]\)

**Parametric:**
\(\text{CVaR}_\alpha = -\bar{r} + \sigma \cdot \phi(z_\alpha) / \alpha\)

**Citation:** Rockafellar & Uryasev (2000); Acerbi & Tasche (2002).
Convention parameter: `method` (`"historical"`, `"parametric"`),
`confidence` (default `0.95`).

### 2.11 Tail Ratio

\[\text{TR}_\alpha = \frac{\mathbb{E}[r_t \mid r_t \geq q_{1-\alpha}]}
{|\mathbb{E}[r_t \mid r_t \leq q_\alpha]|}\]

Ratio of mean upper-tail return to absolute mean lower-tail return.

**Citation:** Connor, Goldberg & Korajczyk (2010, *Portfolio Risk Analysis*, Ch. 9).
Convention parameter: `tail_cutoff` — default `0.05`.

### 2.12 Common-Sense Ratio

\[\text{CSR} = \text{TR}_\alpha \times
\frac{\sum\max(r_t,0)}{|\sum\min(r_t,0)|}
= \text{TR}_\alpha \times \text{Gain-to-Pain Ratio}\]

**Citation:** Bacon (2008, §7.5).

### 2.13 Hill Tail Index (EVT)

\[\hat{\xi} = \frac{1}{k}\sum_{i=1}^{k}
\ln\frac{X_{(i)}}{X_{(k+1)}}\]

where \(X_{(i)}\) are descending order statistics of the upper tail and
\(k\) = `tail_fraction` × \(n\).

**Citation:** Hill (1975). Convention parameter: `tail_fraction` — default
`0.10`.

### 2.14 GPD Tail Fit

Fit the Generalized Pareto Distribution to exceedances above threshold
\(u\) (90th percentile of negative returns):

\[G_{\xi,\beta}(x) = 1 - \left(1 + \xi\frac{x}{\beta}\right)^{\!-1/\xi}\]

**Citation:** Pickands (1975); Hosking & Wallis (1987); Embrechts,
Klüppelberg & Mikosch (1997).

### 2.15 Risk of Ruin

\[P_{\text{ruin}} = \Phi\!\left(
\frac{-\bar{r} \cdot T}{\sigma\sqrt{T}}\right)\]

Normal-approximation probability of 100 % loss. **⚠ Caveat:** assumes
normality — unreliable for fat-tailed returns. Docstring must cite and
warn.

**Citation:** Standard risk management; Vince (1990).

### 2.16 Drawdown Volatility

\[\sigma_{\text{DD}} = \text{std}(d_1, \dots, d_n)\]

where \(d_t = (P_t - \max_{\tau \leq t}P_\tau) / \max_{\tau \leq t}P_\tau\)
is the drawdown series.

**Citation:** Standard risk metric.

### 2.17 Drawdown Periods Count

\[K = \text{number of distinct drawdown episodes}\]

An episode begins when \(P_t\) falls below the running maximum and ends
when it returns to the running maximum.

**Citation:** Standard drawdown analysis.

### 2.18 Current Drawdown

\[d_{\text{current}} = \frac{P_n - \max_{\tau \leq n}P_\tau}
{\max_{\tau \leq n}P_\tau}\]

Drawdown at the last observation.

**Citation:** Standard risk metric.

### 2.19 Current Drawdown Duration

Periods elapsed from the most recent peak to the current observation.

**Citation:** Standard risk metric.

### 2.20 Drawdown Total Duration

\[T_{\text{DD}}^{\text{total}} = \sum_{k=1}^{K} T_{\text{DD}}^{(k)}\]

Sum of all underwater-period lengths.

**Citation:** Standard risk metric.

---

## 3. `core.returns` — Risk-Adjusted

Tagged `("risk_adjusted", "returns")`, backend `"vectorized"`.

### 3.1 Sharpe Ratio

\[\text{SR} = \frac{\bar{r}_{\text{excess}}}{\sigma} \cdot \sqrt{P}\]

where \(\bar{r}_{\text{excess}} = \bar{r} - r_f\) (risk-free rate, default
0).

**Citation:** Sharpe (1966, 1994). Convention parameter: `ddof` — `1`
(default, sample std) or `0` (population std).

### 3.2 Sortino Ratio

\[\text{Sortino} = \frac{\bar{r}_{\text{excess}} \cdot P}
{\text{DD} \cdot \sqrt{P}}\]

where DD is downside deviation (§2.7).

**Citation:** Sortino & Price (1994). Convention parameter: `denominator` —
`"full_downside"` (default) or `"downside_only"`.

### 3.3 Calmar Ratio

\[\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}\]

**Citation:** Young (1991).

### 3.4 Omega Ratio

\[\Omega(\tau) = \frac{\sum_{t=1}^{n}\max(r_t - \tau,\, 0)}
{|\sum_{t=1}^{n}\min(r_t - \tau,\, 0)|}\]

**Citation:** Keating & Shadwick (2002). Default threshold \(\tau = 0\).

### 3.5 Sterling Ratio

\[\text{Sterling} = \frac{\text{CAGR}}{|\text{ADD}| + k}\]

where \(k = 0.10\) avoids near-zero denominators.

**Citation:** Industry convention; Bacon (2008, §8.3).

### 3.6 Burke Ratio

\[\text{Burke} = \frac{\bar{r}_{\text{excess}}}
{\sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(r_t, 0)^2}}\]

**Citation:** Bacon (2008, §8.5).

### 3.7 Kappa-3

\[\text{Kappa}_3(\tau) = \frac{\bar{r} - \tau}
{\sqrt[3]{\frac{1}{n}\sum_{t=1}^{n}\max(\tau - r_t,\; 0)^3}}\]

Lower partial moment of order 3.

**Citation:** Kaplan & Knowles (2004). Convention parameter: `mar` —
default `0.0`.

### 3.8 Martin Ratio (Return / Ulcer)

\[\text{Martin} = \frac{\bar{r}_{\text{excess}}}{\text{UI}}\]

**Citation:** Martin & McCann (1989); Bacon (2008, §8.6).

### 3.9 Gain-to-Pain Ratio

\[\text{GPR} = \frac{\sum_{t=1}^{n}\max(r_t, 0)}
{|\sum_{t=1}^{n}\min(r_t, 0)|}\]

**Citation:** Bacon (2008, §8.4).

---

## 4. `core.returns` — Inference

Tagged `("inference", "returns")`, backend `"resampling"`.

### 4.1 Jarque-Bera Statistic

\[\text{JB} = \frac{n}{6}\!\left(\gamma_1^2 +
\frac{(\gamma_2 + 3 - 3)^2}{4}\right)
= \frac{n}{6}\!\left(\gamma_1^2 + \frac{\gamma_2^2}{4}\right)\]

Under the null of normality, \(\text{JB} \sim \chi^2(2)\).

**Citation:** Jarque & Bera (1980, 1987).

### 4.2 Probabilistic Sharpe Ratio (PSR)

\[\text{PSR}(SR^*) = \Phi\!\left(
\frac{(\hat{SR} - SR^*)\sqrt{T-1}}
{\sqrt{1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}\right)\]

Probability that true SR exceeds benchmark \(SR^*\), adjusted for skewness
and kurtosis.

**Citation:** Bailey & López de Prado (2012); de Prado (2018, Ch. 14).
Convention parameter: `sharpe_benchmark` — default `0.0`.

### 4.3 Deflated Sharpe Ratio (DSR)

\[\text{DSR} = \Phi\!\left(
\frac{(\hat{SR} - SR^*)\sqrt{T-1}}
{\sqrt{1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}
\cdot \left(1 - \frac{1}{M}\sum_{m=1}^{M}
\mathbf{1}_{[\text{SR}_m^{(b)} < \hat{SR}]}\right)\!\right)\]

PSR deflated by the probability of multiple-testing false discovery among
\(M\) trials.

**Citation:** Bailey & López de Prado (2014).

### 4.4 Lo's Autocorrelation-Adjusted Sharpe SE

\[\text{SE}_{\text{IID}}(\hat{SR}) = \sqrt{\frac{1}{T}
\!\left(1 + \frac{1}{2}\hat{SR}^2 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-3}{4}\hat{SR}^2\right)}\]

\[\text{SE}_{\text{adj}} = \text{SE}_{\text{IID}} \cdot
\sqrt{\frac{1+\rho_1}{1-\rho_1}}\]

**Citation:** Lo (2002).

### 4.5 Sharpe Ratio CI — Analytic

\[\hat{SR} \pm z_{1-\alpha/2} \cdot \text{SE}(\hat{SR})\]

using Lo's SE (IID or adjusted per §4.4).

**Citation:** Lo (2002).

### 4.6 Sharpe Ratio CI — Bootstrap

Non-parametric block bootstrap (BCa method): resample overlapping blocks,
recompute SR, form CI from the bootstrap distribution.

**Citation:** Efron & Tibshirani (1994); Ledoit & Wolf (2008); block-length
selection per Politis & White (2004). Default: 5,000 replications,
automatic block length, 95 % CI.

### 4.7 Minimum Track Record Length

\[T_{\min} = \left(\frac{z_{1-\alpha}}{SR^* - \hat{SR}}\right)^{\!2}
\cdot \left(1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2\right)\]

**Citation:** Bailey & López de Prado (2012); de Prado (2018, Ch. 14).
Default \(\alpha = 0.05\).

### 4.8 Generic Block-Bootstrap CI Wrapper

Moving-block bootstrap: for any registered `returns`-tier metric function
\(f\), resample blocks of length \(l\) with replacement, recompute \(f\) on
each resample, return empirical CI.

**Citation:** Efron & Tibshirani (1994); Künsch (1989); Politis & White
(2004). Default: 5,000 reps, automatic block length, 95 % equal-tailed CI.

---

## 5. `core.returns` — Rolling / Regime Wrappers

Generic functions applied via the registry to any `returns`-tier metric.

### 5.1 `rolling(metric_name, window)`

Apply the named metric over a rolling window of `window` periods,
producing a time series of that metric.

**Citation:** Standard rolling-window analysis.

### 5.2 `by_regime(metric_name, regime_labels)`

Group returns by regime label (same-length array), compute the named
metric separately per regime.

**Citation:** Ang & Bekaert (2004).

---

## 6. `core.exposure`

Tagged `requires="exposure"`. Category tags vary.

### 6.1 Gross Exposure

\[\text{GE}_t = \sum_{i=1}^{N} \left|\frac{\text{position\_value}_{i,t}}
{\text{portfolio\_value}_t}\right|\]

Report: current, max, average.

**Citation:** Ang (2014, *Asset Management*, Ch. 2).

### 6.2 Net Exposure

\[\text{NE}_t = \sum_{i=1}^{N} \frac{\text{position\_value}_{i,t}}
{\text{portfolio\_value}_t}\]

Report: current, max, min, average, range.

**Citation:** Same as above.

### 6.3 Leverage

\[\text{Leverage}_t = \frac{\text{gross\_exposure}_t}{\text{equity}_t}\]

Ratio of gross exposure to equity.

**Citation:** Ang (2014, Ch. 2).

### 6.4 Long Exposure %

\[\text{LE}\%_t = \sum_{i} w_{i,t} \cdot \mathbf{1}_{[w_{i,t} > 0]}\]

Fraction of gross exposure allocated long.

**Citation:** Standard analytics.

### 6.5 Short Exposure %

\[\text{SE}\%_t = \sum_{i} |w_{i,t}| \cdot \mathbf{1}_{[w_{i,t} < 0]}\]

Fraction of gross exposure allocated short.

**Citation:** Standard analytics.

### 6.6 Long-Book Contribution to Return

\[R_t^{\text{long}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
\mathbf{1}_{[w_{i,t-1} > 0]}\]

**Citation:** Practitioner P&L attribution.

### 6.7 Short-Book Contribution to Return

\[R_t^{\text{short}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
\mathbf{1}_{[w_{i,t-1} < 0]}\]

**Citation:** Practitioner P&L attribution.

### 6.8 Long Beta

\[\beta_{\text{long}} = \frac{\text{Cov}(R^{\text{long}}, R_m)}
{\text{Var}(R_m)}\]

**Requires:** optional benchmark field on `ExposureInput`.

**Citation:** Asness, Frazzini & Pedersen (2014).

### 6.9 Short Beta

\[\beta_{\text{short}} = \frac{\text{Cov}(R^{\text{short}}, R_m)}
{\text{Var}(R_m)}\]

**Requires:** optional benchmark field on `ExposureInput`.

**Citation:** Same as above.

### 6.10 Position Concentration (HHI)

\[\text{HHI}_t = \sum_{i=1}^{N}
\left(\frac{w_{i,t}}{\sum_j |w_{j,t}|}\right)^{\!2}\]

Herfindahl-Hirschman Index of normalized position weights.

**Citation:** Hirschman (1964); SEC concentration reporting.

### 6.11 Effective N Positions

\[N_{\text{eff},t} = \frac{1}{\text{HHI}_t}\]

Reciprocal HHI.

**Citation:** Standard concentration/effective-count duality.

### 6.12 Turnover

\[\text{TO}_t = \frac{1}{2}\sum_{i=1}^{N}
|\Delta w_{i,t}|\]

Annualized: mean period turnover × \(P\).

**Citation:** Industry standard; Morningstar methodology.

### 6.13 Average Holding Weight per Position

\[\bar{w}_t = \frac{1}{N_t}\sum_{i=1}^{N_t} |w_{i,t}|\]

**Citation:** Standard analytics.

### 6.14 Position Coverage %

\[\text{Coverage} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[\exists i: w_{i,t} \neq 0]}\]

Fraction of periods with any position.

**Citation:** Standard analytics.

### 6.15 Long/Short Position Coverage %

Fraction of periods with at least one long (short) position.

**Citation:** Standard analytics.

### 6.16 Exposure Volatility

\[\sigma_{\text{GE}} = \text{std}(\text{GE}_1, \dots, \text{GE}_n)\]

Standard deviation of gross exposure.

**Citation:** Standard risk metric.

### 6.17 Net Exposure Volatility

\[\sigma_{\text{NE}} = \text{std}(\text{NE}_1, \dots, \text{NE}_n)\]

**Citation:** Standard risk metric.

### 6.18 Exposure Coefficient of Variation

\[\text{CV}_{\text{exp}} =
\frac{\sigma_{\text{GE}}}{|\bar{\text{GE}}|}\]

**Citation:** Standard statistics.

### 6.19 Avg Exposure Utilization

\[\text{Utilization} = \frac{\bar{\text{GE}}}{\max_t \text{GE}_t}\]

Mean gross exposure relative to maximum.

**Citation:** Standard analytics.

### 6.20 Exposure Directional Bias

\[\text{Bias} = \frac{|\bar{\text{NE}}|}{\bar{\text{GE}}}\]

Absolute mean net exposure relative to mean gross exposure.

**Citation:** Standard analytics.

### 6.21 Exposure Percentiles

Percentiles at {25, 50, 75, 90, 95} of the gross exposure time series.

**Citation:** Standard order statistics.

### 6.22 Period Counts

Breakdown: total periods, periods with any position, long-only periods,
short-only periods, idle (flat) periods.

**Citation:** Standard analytics.

---

## 7. `core.trades`

Tagged `requires="trades"`. Category tags vary.

### 7.1 Total Trades

\[N = \text{number of round-trip trades}\]

### 7.2 Win Rate (Overall)

\[\text{WR} = \frac{N_{\text{win}}}{N}\]

**Citation:** Standard trade analysis.

### 7.3 Win Rate (Long-Only)

Same formula, restricted to long trades.

### 7.4 Win Rate (Short-Only)

Same formula, restricted to short trades.

### 7.5 Average Win

\[\bar{W} = \frac{1}{N_{\text{win}}}\sum_{j:\text{win}}\text{PnL}_j\]

**Citation:** Standard trade analysis.

### 7.6 Average Loss

\[\bar{L} = \frac{1}{N_{\text{loss}}}\sum_{j:\text{loss}}\text{PnL}_j\]

Report absolute value; sign preserved in return object.

**Citation:** Standard trade analysis.

### 7.7 Win/Loss Ratio

\[\text{WLR} = \frac{N_{\text{win}}}{N_{\text{loss}}}\]

### 7.8 Profit Factor

\[\text{PF} = \frac{\sum_{j}\max(\text{PnL}_j, 0)}
{|\sum_{j}\min(\text{PnL}_j, 0)|}\]

**Citation:** Schwager (1995).

### 7.9 Expectancy per Trade

\[\mathbb{E}[\text{PnL}] = \text{WR} \cdot \bar{W} +
(1 - \text{WR}) \cdot \bar{L}\]

with \(\bar{L}\) as a negative number.

### 7.10 Average Holding Period

\[\bar{H} = \frac{1}{N}\sum_{j=1}^{N}(t_{j,\text{exit}} -
t_{j,\text{entry}})\]

### 7.11 Holding Period Distribution

Median, 25th, 75th percentile, min, max of holding periods.

### 7.12 Max Consecutive Wins

Longest run of consecutive winning trades.

### 7.13 Max Consecutive Losses

Longest run of consecutive losing trades.

### 7.14 Round-Trip P&L Distribution

Summary statistics: mean, median, std, skewness, 5th/95th percentiles
of per-trade P&L.

### 7.15 Implementation Shortfall

\[\text{IS}_j = \text{side}_j \cdot
\frac{P_{\text{fill}} - P_{\text{decision}}}
{P_{\text{decision}}}\]

**Requires:** optional fill-price field.

**Citation:** Perold (1988).

### 7.16 Best Trade

\[\max_j \text{PnL}_j\]

### 7.17 Worst Trade

\[\min_j \text{PnL}_j\]

### 7.18 Avg Winning Trade Duration

\[\bar{H}_{\text{win}} = \frac{1}{N_{\text{win}}}
\sum_{j:\text{win}}(t_{j,\text{exit}} - t_{j,\text{entry}})\]

### 7.19 Avg Losing Trade Duration

\[\bar{H}_{\text{loss}} = \frac{1}{N_{\text{loss}}}
\sum_{j:\text{loss}}(t_{j,\text{exit}} - t_{j,\text{entry}})\]

### 7.20 Payoff Ratio

\[\text{Payoff} = \frac{\bar{W}}{|\bar{L}|}\]

Average win divided by absolute average loss.

**Citation:** Standard trade analysis.

### 7.21 CPC Ratio

\[\text{CPC} = \text{PF} \times \text{Payoff} \times \text{WR}\]

Young's CPC Index: profit factor × payoff ratio × win rate.

**Citation:** Young (1991), CPC Index.

### 7.22 SQN (System Quality Number)

\[\text{SQN} = \frac{\bar{r}_{\text{trade}}}
{\sigma_{\text{trade}}} \cdot \sqrt{N}\]

Mean trade return divided by trade return std, scaled by √(n trades).

**Citation:** Tharp (1998, *Trade Your Way to Financial Freedom*).

### 7.23 Trade Duration Std

\[\sigma_H = \text{std}(H_1, \dots, H_N)\]

### 7.24 Trade Return Std

\[\sigma_{\text{trade}} = \text{std}(\text{PnL}_1, \dots, \text{PnL}_N)\]

### 7.25 Geometric Mean Return (per Trade)

\[\bar{r}_{g,\text{trade}} = \exp\!\left(
\frac{1}{N}\sum_{j=1}^{N}\ln(1 + \text{PnL}_j)\right) - 1\]

### 7.26 Outlier Win Ratio

Fraction of winning trades with P&L exceed Q₃ + 1.5 × IQR.

**Citation:** Tukey (1977) outlier detection.

### 7.27 Outlier Loss Ratio

Fraction of losing trades with P&L below Q₁ − 1.5 × IQR.

### 7.28 MFE (Maximum Favorable Excursion)

Maximum dollar gain during trade lifetime.

**Requires:** optional intra-trade price path field.

**Citation:** Standard trade analytics.

### 7.29 MAE (Maximum Adverse Excursion)

Maximum dollar loss during trade lifetime.

**Requires:** optional intra-trade price path field.

**Citation:** Standard trade analytics.

### 7.30 Kelly Criterion

\[f^* = W - \frac{1-W}{\bar{W}/|\bar{L}|}\]

Optimal bet size.

**Citation:** Kelly (1956); Thorp (1997).

### 7.31 Long/Short Trade Count

\(N_{\text{long}}\), \(N_{\text{short}}\).

### 7.32 Long/Short Trade %

\(N_{\text{long}} / N\), \(N_{\text{short}} / N\).

### 7.33 Long/Short Winning/Losing Trades

Win/loss breakdown by side.

### 7.34 Long/Short Avg Duration

\(\bar{H}_{\text{long}}\), \(\bar{H}_{\text{short}}\).

### 7.35 Long/Short Total PnL %

Total P&L attributable to long and short trades.

### 7.36 Long/Short Avg PnL %

### 7.37 Long/Short Best/Worst Trade %

---

## 8. `core.benchmark`

Tagged `requires="benchmark"`. Category tags vary.

### 8.1 Alpha (Jensen's)

\[\alpha_{\text{ann}} = \text{CAGR} - (r_f + \beta \cdot
(\text{CAGR}_m - r_f))\]

**Citation:** Jensen (1968).

### 8.2 Beta

\[\beta = \frac{\text{Cov}(r, r_m)}{\text{Var}(r_m)}\]

**Citation:** Sharpe (1964). Convention parameter: `variant` —
`"least_squares"` (default).

### 8.3 R²

\[R^2 = 1 - \frac{\text{Var}(r - \hat{r})}{\text{Var}(r)}\]

where \(\hat{r} = \alpha + \beta r_m\).

**Citation:** Standard OLS goodness-of-fit.

### 8.4 Tracking Error

\[\text{TE}_{\text{ann}} = \sigma(r - r_m) \cdot \sqrt{P}\]

**Citation:** Roll (1992).

### 8.5 Information Ratio

\[\text{IR}_{\text{ann}} = \frac{(\bar{r} - \bar{r}_m) \cdot P}
{\sigma(r - r_m) \cdot \sqrt{P}}\]

**Citation:** Goodwin (1998).

### 8.6 Up-Capture Ratio

\[\text{UC} = \frac{\text{mean}_{t: r_{m,t} > 0}(r_t)}
{\text{mean}_{t: r_{m,t} > 0}(r_{m,t})}\]

**Citation:** Morningstar methodology.

### 8.7 Down-Capture Ratio

\[\text{DC} = \frac{\text{mean}_{t: r_{m,t} < 0}(r_t)}
{\text{mean}_{t: r_{m,t} < 0}(r_{m,t})}\]

**Citation:** Same as above.

### 8.8 Up/Down Capture Ratio

\[\text{UDR} = \frac{\text{UC}}{|\text{DC}|}\]

### 8.9 Correlation

\[\rho = \frac{\text{Cov}(r, r_m)}{\sigma(r)\cdot\sigma(r_m)}\]

Pearson correlation with benchmark.

### 8.10 Active Return

\[\bar{r}_{\text{active, ann}} = (\bar{r} - \bar{r}_m) \cdot P\]

### 8.11 Batting Average vs Benchmark

\[\text{BA} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[r_t > r_{m,t}]}\]

### 8.12 Treynor Ratio

\[\text{Treynor} = \frac{\bar{r}_{\text{excess}} \cdot P}{\beta}\]

**Citation:** Treynor (1965).

### 8.13 Outperformance

\[R_{\text{out}} = R_{\text{cum}} - R_{m,\text{cum}}\]

### 8.14 Outperformance Ratio

\[\text{OR} = \frac{1 + R_{\text{cum}}}{1 + R_{m,\text{cum}}}\]

### 8.15 Underperforming Periods / %

Count and fraction of periods where \(r_t < r_{m,t}\).

### 8.16 Max Outperformance

Maximum cumulative outperformance (active return) over the full period.

### 8.17 Max Underperformance

Maximum cumulative underperformance (absolute value) over the full period.

### 8.18 Benchmark Volatility

\[\sigma_{m,\text{ann}} = \sigma(r_m) \cdot \sqrt{P}\]

---

## 9. `core.compare`

Tagged `requires="compare"`, `category=("relative", "compare")`.

### 9.1 Correlation Matrix

\[\Sigma_{ij} = \text{Corr}(r^{(i)}, r^{(j)})\]
for \(i,j = 1,\dots,K\).

**Citation:** Standard multivariate statistics.

### 9.2 Diversification Ratio

\[\text{DR} = \frac{\sum_{i=1}^{K} w_i\sigma_i}{\sigma_p}\]

with equal-weight (default) or user-specified weights.

**Citation:** Choueifaty & Coignard (2008).

### 9.3 Pairwise Sharpe-Difference Test (Jobson-Korkie, Memmel)

\[z = \frac{\hat{SR}_1 - \hat{SR}_2}{\sqrt{\hat{\sigma}^2}}
\sim \mathcal{N}(0,1)\]

\[\hat{\sigma}^2 = \frac{1}{T}\!\left[2 - 2\rho_{12} +
\frac{1}{2}(\hat{SR}_1^2 + \hat{SR}_2^2) -
\rho_{12}^2\hat{SR}_1\hat{SR}_2 -
(\gamma_{3,1}\hat{SR}_1 - \gamma_{3,2}\hat{SR}_2)\cdot 2\rho_{12}\right]\]

**Citation:** Jobson & Korkie (1981); Memmel (2003).

### 9.4 White's Reality Check / SPA Test

White's RC: \(\bar{V} = \max_k \sqrt{T} \cdot \bar{f}_k\), bootstrap
distribution of the max statistic. SPA (Hansen 2005) studentizes it.

**Citation:** White (2000); Hansen (2005).

### 9.5 PBO (Combinatorial Purged CV)

Generate combinatorially-paired train/test splits with purge + embargo;
PBO = probability that the best in-sample strategy ranks below median
out-of-sample.

**Citation:** Bailey & López de Prado (2014); de Prado (2018, Ch. 11–13).

### 9.6 Marginal Contribution to Portfolio Risk

\[\text{MCR}_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}\]

**Citation:** Litterman (1996); Qian (2005).

### 9.7 Component VaR

\[\text{CVaR}_i = w_i \cdot \frac{\partial \text{VaR}}{\partial w_i}
= w_i \cdot (-\beta_i \cdot \text{VaR}_p)\]

**Citation:** Jorion (2006, *Value at Risk*, 3rd ed., Ch. 7).

---

## 10. `report`

Separate install extra (`stratstat[report]`). Depends on `core` only — never
the reverse. No formulas; specification of visualization outputs.

| # | Output | Description |
|---|--------|-------------|
| 1 | Single-strategy tear sheet | Equity curve, drawdown chart with recovery shading, monthly heatmap, stats table |
| 2 | Multi-strategy comparison dashboard | Overlay equity curves, rolling-metric charts, correlation heatmap, ranking table |
| 3 | Rolling-metric charts | Any registered metric via `rolling()`, rendered as time series |
| 4 | Drawdown chart | Underwater equity curve with recovery-period shading |
| 5 | Cumulative return chart | Cumulative return index, optionally with benchmark overlay |
| 6 | Benchmark comparison overlay | Strategy vs. benchmark cumulative return and active return |
| 7 | Trade markers (MFE/MAE) | Optional intra-trade MFE/MAE overlay on equity curve |
| 8 | Monthly heatmap matrix | Years × months return grid with color scale |
| 9 | Export formats | Interactive HTML, static PNG/SVG, Markdown table, LaTeX table, JSON |

---

## References

1. Acerbi, C. & Tasche, D. (2002). "On the Coherence of Expected Shortfall." *JBF*, 26(7).
2. Ang, A. (2014). *Asset Management*. Oxford University Press.
3. Artzner, P. et al. (1999). "Coherent Measures of Risk." *Math. Finance*, 9(3).
4. Bacon, C. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd ed. Wiley.
5. Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *J. Risk*, 15(2).
6. — (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management*, 40(5).
7. Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton.
8. Choueifaty, Y. & Coignard, Y. (2008). "Toward Maximum Diversification." *J. Portfolio Management*, 35(1).
9. de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
10. Efron, B. & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall.
11. Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
12. Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *JBES*, 23(4).
13. Hill, B. M. (1975). "A Simple General Approach to Inference About the Tail." *Ann. Stat.*, 3(5).
14. Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter Estimation for the GPD." *Technometrics*, 29(3).
15. Jarque, C. M. & Bera, A. K. (1987). "A Test for Normality." *Int. Stat. Rev.*, 55(2).
16. Jensen, M. C. (1968). "The Performance of Mutual Funds 1945–1964." *J. Finance*, 23(2).
17. Jobson, J. D. & Korkie, B. M. (1981). "Performance Hypothesis Testing with Sharpe." *J. Finance*, 36(4).
18. Jorion, P. (2006). *Value at Risk*, 3rd ed. McGraw-Hill.
19. Kaplan, P. D. & Knowles, J. A. (2004). "Kappa: A Generalized Downside Risk-Adjusted Measure." *JPM*.
20. Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *JPM*, 6(3).
21. Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Tech. J.*, 35(4).
22. Künsch, H. R. (1989). "The Jackknife and the Bootstrap." *Ann. Stat.*, 17(3).
23. Ledoit, O. & Wolf, M. (2008). "Robust Performance Hypothesis Testing." *J. Empirical Finance*, 15(5).
24. Litterman, R. (1996). "Hot Spots and Hedges." *JPM*, Special Issue.
25. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4).
26. Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
27. Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio." *Finance Letters*, 1(1).
28. Perold, A. F. (1988). "The Implementation Shortfall." *JPM*, 14(3).
29. Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Ann. Stat.*, 3(1).
30. Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection." *Econometric Reviews*, 23(1).
31. Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *J. Risk*, 2(3).
32. Roll, R. (1992). "A Mean/Variance Analysis of Tracking Error." *JPM*, 18(4).
33. Schwager, J. D. (1995). *Schwager on Futures: Technical Analysis*. Wiley.
34. Sharpe, W. F. (1964). "Capital Asset Prices." *J. Finance*, 19(3).
35. — (1966). "Mutual Fund Performance." *J. Business*, 39(1).
36. — (1994). "The Sharpe Ratio." *JPM*, 21(1).
37. Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *J. Investing*, 3(3).
38. Tharp, V. K. (1998). *Trade Your Way to Financial Freedom*. McGraw-Hill.
39. Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *Harvard Business Review*, 43(1).
40. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
41. White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5).
42. Young, T. W. (1991). "Calmar Ratio: A Smoother Tool." *Futures*, 20(10).
43. Zangari, P. (1996). "A VaR Methodology for Portfolios That Include Options." *RiskMetrics Monitor*.
