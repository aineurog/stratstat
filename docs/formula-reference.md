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
| Drawdown duration | `units` | `periods`, `years` | `periods` | van Hemert et al. (2020, Ch. 5) |
| VaR | `method` | `historical`, `parametric`, `cornish_fisher` | `historical` | Litterman (1996); Zangari (1996) |
| VaR / CVaR | `confidence` | `float` in `(0, 1)` | `0.95` | Jorion (2006, §5.2); Basel Committee (1996) |
| CVaR / ES | `method` | `historical`, `parametric` | `historical` | Rockafellar & Uryasev (2000) |
| Beta | `variant` | `least_squares` | `least_squares` | Sharpe (1964) |
| Annualized volatility | `return_type` | `simple`, `log` | `simple` | CFA Institute, *Quantitative Methods* |
| Tail ratio | `tail_cutoff` | `float` in `(0, 0.5)` | `0.05` | Connor, Goldberg & Korajczyk (2010, Ch. 9) |
| Hill tail index | `tail_fraction` | `float` in `(0, 1)` | `0.10` | Hill (1975); upper 10 % tail |
| PSR | `sharpe_benchmark` | `float` ≥ 0 | `0.0` | Bailey & López de Prado (2012) |
| Kappa-3 | `mar` | float (minimum acceptable return) | `0.0` | Kaplan & Knowles (2004) |
| Sterling ratio | `floor` | `float` ≥ 0 | `0.10` | Bacon (2008, §8.3) |

**Note on annualization:** `periods_per_year` is an input-level parameter
set on the input container (e.g., `ReturnsInput`). It is not a per-metric
convention. Every metric that supports annualization reads
`periods_per_year` from the input container. The Sharpe ratio's
"annualization convention" noted in `stats.md` refers to whether
multiplication by √P happens before or after dividing by σ — the formula
reference uses the standard pre-division form (Sharpe 1994).

---

## 1. `core.returns` — Descriptive

All tagged `("descriptive", "returns")`, backend `"vectorized"`.

### 1.1 CAGR (Compound Annual Growth Rate)

$$\text{CAGR} = \exp\!\left(\frac{1}{T}\sum_{t=1}^{n}\ln(1+r_t)\right) - 1$$

where $T$ is the length of the series in years. Equivalent to
$(V_f/V_i)^{1/T} - 1$.

**Citation:** Damodaran (2012, *Investment Valuation*, 3rd ed., Ch. 3).

### 1.2 Annualized Volatility

$$\sigma_{\text{ann}} = \sigma \cdot \sqrt{P}$$

where $\sigma$ is the sample standard deviation (ddof = 1) of period
returns and $P$ is `periods_per_year`.

With `return_type="log"`, $\sigma$ is the standard deviation of the log
returns $\ln(1 + r_t)$ instead of the simple returns $r_t$. This is the
StratStat equivalent of QuantStats `implied_volatility`.

**Citation:** CFA Institute, *Quantitative Methods* (CFA Program
Curriculum, Level I, Vol. 1). Convention parameter: `return_type` —
`"simple"` (default) or `"log"`.

### 1.3 Cumulative Return

$$R_{\text{cum}} = \prod_{t=1}^{n}(1 + r_t) - 1$$

**Citation:** Bacon (2008, *Practical Portfolio Performance Measurement
and Attribution*, 2nd ed., §2.1). Elementary computation; the citation
establishes the standard compounding convention.

### 1.4 Arithmetic Mean Return

$$\bar{r} = \frac{1}{n}\sum_{t=1}^{n} r_t$$

**Citation:** No single canonical source — elementary statistic. The
sample mean is unbiased under IID returns; see any introductory statistics
text (e.g., Casella & Berger 2002, *Statistical Inference*, §5.2).

### 1.5 Geometric Mean Return

$$\bar{r}_g = \left(\prod_{t=1}^{n}(1 + r_t)\right)^{\!1/n} - 1$$

**Citation:** Campbell, Lo & MacKinlay (1997, *The Econometrics of
Financial Markets*, §1.4).

### 1.6 Skewness

$$\gamma_1 = \frac{n}{(n-1)(n-2)}\sum_{t=1}^{n}\!\left(\frac{r_t -
\bar{r}}{\sigma}\right)^{\!3}$$

Adjusted (sample) skewness.

**Citation:** Fisher (1930), "The Moments of the Distribution for Normal
Samples of Measures of Departure from Normality." Also Cramér (1946,
*Mathematical Methods of Statistics*, §27.4).

### 1.7 Excess Kurtosis

$$\gamma_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}
\sum_{t=1}^{n}\!\left(\frac{r_t - \bar{r}}{\sigma}\right)^{\!4}
- \frac{3(n-1)^2}{(n-2)(n-3)}$$

Returns 0 for a normal distribution (`fisher=True`).

**Citation:** Fisher (1930); Cramér (1946, §27.4).

### 1.8 Best Period

$$r_{\text{best}} = \max_t r_t$$

At the data's native frequency.

**Citation:** Elementary order statistic. Bacon (2008, §3.10) documents
best/worst-period as standard performance analytics.

### 1.9 Worst Period

$$r_{\text{worst}} = \min_t r_t$$

**Citation:** As §1.8. Bacon (2008, §3.10).

### 1.10 Positive-Period Ratio

$$\text{PPR} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}_{[r_t > 0]}$$

Strictly positive (> 0); zero is non-positive.

**Citation:** Bacon (2008, §3.11).

### 1.11 Autocorrelation (Lag-1)

$$\rho_1 = \frac{\sum_{t=2}^{n}(r_t - \bar{r})(r_{t-1} - \bar{r})}
{\sum_{t=1}^{n}(r_t - \bar{r})^2}$$

**Citation:** Campbell, Lo & MacKinlay (1997, §2.4).

### 1.12 Variance

$$s^2 = \frac{1}{n-1}\sum_{t=1}^{n}(r_t - \bar{r})^2$$

Sample variance (ddof = 1).

**Citation:** Elementary statistic. Fisher (1925, *Statistical Methods for
Research Workers*) established n−1 degrees-of-freedom correction.

### 1.13 Return Range

$$R_{\text{range}} = \max_t r_t - \min_t r_t$$

**Citation:** Elementary order statistic. Bacon (2008, §3.10) uses range
in the context of period-return dispersion.

### 1.14 Percentiles

The $p$-th percentile at levels \(p \in \{1, 5, 10, 25, 50, 75, 90,
95, 99\}\) of the empirical return distribution, via linear interpolation
(Hyndman & Fan 1996, type 7).

**Citation:** Hyndman & Fan (1996), "Sample Quantiles in Statistical
Packages," *The American Statistician*, 50(4).

### 1.15 Coefficient of Variation

$$\text{CV} = \frac{\sigma}{|\bar{r}|}$$

**Citation:** Pearson (1896), "Mathematical Contributions to the Theory of
Evolution. III. Regression, Heredity, and Panmixia." Also documented in
Everitt & Skrondal (2010, *The Cambridge Dictionary of Statistics*).

### 1.16 Outlier Count & % (IQR Method)

$$r_t \text{ is an outlier if } r_t < Q_1 - 1.5 \times \text{IQR}
\;\text{ or }\; r_t > Q_3 + 1.5 \times \text{IQR}$$

where $\text{IQR} = Q_3 - Q_1$.

**Citation:** Tukey (1977, *Exploratory Data Analysis*).

### 1.17 Stability of Time Series

$$\text{Stability} = R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$

from the OLS regression of the cumulative log return on time:

$$y_t = \sum_{\tau=1}^{t}\ln(1 + r_\tau), \qquad
y_t = \alpha + \beta\,t + \varepsilon_t$$

An $R^2$ close to 1 indicates a smooth, consistent growth path; close to 0
indicates an erratic equity curve. NaN for fewer than 3 observations.

**Citation:** Standard regression; cited in empyrical.

### 1.18 Hurst Exponent

Rescaled-range (R/S) analysis. For lag scales
$\tau = \lfloor n / 2^k \rfloor$ ($k = 1, 2, \dots$, stopping when
$\tau < 10$), partition the series into blocks of length $\tau$. For each
block compute the cumulative deviate from the block mean, then

$$(R/S)_\tau = \text{mean}_{\text{blocks}}
\frac{\max(\text{dev}) - \min(\text{dev})}{\sigma_{\text{block}}}$$

Regressing $\ln(R/S)_\tau$ on $\ln\tau$ gives slope $H$. Requires at least
50 periods. $H > 0.5$: trending; $H \approx 0.5$: random walk; $H < 0.5$:
mean-reverting.

**Citation:** Hurst (1951); Mandelbrot (1972).

### 1.19 Fractal Dimension

$$D = 2 - H$$

where $H$ is the Hurst exponent (§1.18). $D \approx 1.5$ for a random walk,
$D < 1.5$ for a smoother trending curve, $D > 1.5$ for a rougher
mean-reverting curve. Requires at least 50 periods (inherited from the
Hurst computation).

**Citation:** Mandelbrot (1975).

### 1.20 Consecutive Wins/Losses

Maximum and current streaks of consecutive positive ($r_t > 0$) and
negative ($r_t < 0$) return periods. A period with zero return or NaN
breaks all streaks. Returns a dict with keys `max_win_streak`,
`max_loss_streak`, `current_win_streak`, `current_loss_streak`. This is the
returns-level analogue of the trade-level `max_consecutive_wins` /
`max_consecutive_losses` metrics (§7.12–7.13).

**Citation:** Schwager (1995, *Schwager on Futures: Technical Analysis*).

### 1.21 Negative-Period Ratio

$$\text{NPR} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}_{[r_t < 0]}$$

Fraction of periods with strictly negative return; zero is treated as
non-negative. NPR + PPR ≤ 1, with strict inequality when any period
returns exactly zero.

**Citation:** Bacon (2008, §3.11).

### 1.22 Exposure Time

$$\text{exposure} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}_{[r_t \neq 0]}$$

Share of periods the strategy is invested (non-zero return). NaN periods
are counted as not invested. Maps to QuantStats `exposure`.

**Citation:** Industry convention; no independent academic source identified.

### 1.23 Average Up Period

$$\bar{r}_{\text{up}} = \frac{\sum_{t=1}^{n} r_t\,\mathbf{1}_{[r_t > 0]}}
{\sum_{t=1}^{n}\mathbf{1}_{[r_t > 0]}}$$

Mean of the strictly positive period returns. NaN when no positive periods.
Maps to QuantStats `avg_win`.

**Citation:** Industry convention; no independent academic source identified.

### 1.24 Average Down Period

$$\bar{r}_{\text{down}} = \frac{\sum_{t=1}^{n} r_t\,\mathbf{1}_{[r_t < 0]}}
{\sum_{t=1}^{n}\mathbf{1}_{[r_t < 0]}}$$

Mean of the strictly negative period returns, sign preserved (negative).
NaN when no negative periods. Maps to QuantStats `avg_loss`.

**Citation:** Industry convention; no independent academic source identified.

### 1.25 Period Profit Factor

$$\text{PF} = \frac{\sum_{t=1}^{n}\max(r_t,\,0)}
{|\sum_{t=1}^{n}\min(r_t,\,0)|}$$

Returns-level analogue of the trade-level `profit_factor` (§7.8). Inf when
there are gains but no losses; NaN when there are neither.

**Citation:** QuantStats-compatible convention; no independent academic
source identified.

### 1.26 Period Payoff Ratio

$$\text{payoff} = \frac{\bar{r}_{\text{up}}}{|\bar{r}_{\text{down}}|}$$

Returns-level analogue of the trade-level `payoff_ratio` (§7.20). Maps to
QuantStats `payoff_ratio` / `win_loss_ratio` (which QuantStats defines at
the period level as avg_win / |avg_loss|). Inf when there are gains but no
losses.

**Citation:** QuantStats-compatible convention; no independent academic
source identified.

### 1.27 Period Kelly Criterion

$$f^{*} = W - \frac{1 - W}{\text{payoff}},\qquad
W = \frac{n_{\text{up}}}{n_{\text{up}} + n_{\text{down}}}$$

Optimal bet fraction treating each non-zero period as a bet. Zero and NaN
periods are excluded from the win probability. Returns-level analogue of
the trade-level `kelly_criterion` (§7.30).

**Citation:** Kelly (1956), "A New Interpretation of Information Rate,"
*Bell System Technical Journal*, 35(4); period-based adaptation.

---

## 2. `core.returns` — Risk

Tagged `("risk", "returns")`. Backend varies: mostly `"vectorized"`,
drawdown walks are `"sequential"`.

### 2.1 Max Drawdown

**Simple-return method (default):** Build the cumulative return index
$P_t = \prod_{\tau=1}^{t}(1 + r_\tau)$, then:

$$\text{MDD} = \max_{t}\frac{P_t - \max_{\tau \leq t} P_\tau}
{\max_{\tau \leq t} P_\tau}$$

**Log-return method (`return_type="log"`):** Build the cumulative log-return
index \(P_t^{\log} = \exp(\sum_{\tau=1}^{t} \ln(1 + r_\tau)) =
\prod_{\tau=1}^{t}(1 + r_\tau)\) and apply the same peak-to-trough
formula. The two methods produce identical equity curves when the index
is built multiplicatively from simple returns vs. via log-return
compounding. The distinction matters only when the returns themselves
are log returns: in that case, $P_t = \exp(\sum r_\tau^{\log})$ and the
drawdown is computed on the exponentiated index.

**Citation:** Pospisil & Vecer (2011), "Maximum Drawdown of a Brownian
Motion," *Journal of Applied Probability*, 48(3). Convention parameter:
`return_type` — `"simple"` (default) or `"log"`.

### 2.2 Longest Drawdown Duration

$$T_{\text{DD}}^{\max} = \max_k\!\left(\tau_{\text{end}}^{(k)} -
\tau_{\text{start}}^{(k)}\right)$$

Longest contiguous underwater period. A drawdown episode begins when the
equity curve falls below its running maximum and ends when it first
returns to (or exceeds) that maximum.

**Citation:** van Hemert et al. (2020, *Tactical Asset Allocation*, Ch. 5).
Convention parameter: `units` — `"periods"` (default) or `"years"`.

### 2.3 Time to Recovery

For each drawdown episode $k$, the time in periods from
$t_{\text{peak}}$ to the first subsequent period where the cumulative
return reaches or exceeds the running-maximum level at $t_{\text{peak}}$.
Report mean, median, and maximum across all episodes in the series.

**Citation:** Bacon (2008, §7.2), "Drawdown Analysis."

### 2.4 Average Drawdown

$$\text{ADD} = \frac{1}{K}\sum_{k=1}^{K} D^{(k)}$$

where $D^{(k)}$ is the peak-to-trough depth of the $k$-th episode
(as a negative percentage; absolute value used in display).

**Citation:** Bacon (2008, §7.2).

### 2.5 Average Drawdown Duration

$$\bar{T}_{\text{DD}} = \frac{1}{K}\sum_{k=1}^{K} T_{\text{DD}}^{(k)}$$

**Citation:** Bacon (2008, §7.2).

### 2.6 Ulcer Index

$$\text{UI} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}
\left(\frac{P_t - \max_{\tau \leq t} P_\tau}
{\max_{\tau \leq t} P_\tau}\right)^{\!2}}$$

Root-mean-square of percentage drawdowns.

**Citation:** Martin & McCann (1989, *The Investor's Guide to Fidelity
Funds*); Martin (1993).

### 2.7 Downside Deviation

$$\text{DD} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(r_t - \tau,\; 0)^2}$$

With $\tau = 0$ (MAR), this is semi-deviation.

**Citation:** Sortino & van der Meer (1991); Sortino & Price (1994).

### 2.8 Upside Deviation

$$\text{UD} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\max(r_t - \tau,\; 0)^2}$$

Mirror of downside deviation.

**Citation:** Bacon (2008, §6.4) discusses upside/downside semi-variance
as complementary risk measures.

### 2.9 VaR (Value at Risk)

**Historical (default):**
$\text{VaR}_\alpha = -F_r^{-1}(\alpha)$

**Parametric:**
$\text{VaR}_\alpha = -(\bar{r} + z_\alpha \cdot \sigma)$

**Cornish-Fisher:**
Same as parametric with $z_\alpha$ replaced by the CF expansion:
\(z_{\text{CF}} = z_\alpha + \frac{\gamma_1}{6}(z_\alpha^2 - 1) +
\frac{\gamma_2}{24}(z_\alpha^3 - 3z_\alpha) -
\frac{\gamma_1^2}{36}(2z_\alpha^3 - 5z_\alpha)\)

**Citation:** Litterman (1996), "Hot Spots and Hedges," *JPM*; Zangari
(1996), "A VaR Methodology for Portfolios That Include Options,"
*RiskMetrics Monitor*. Basel Committee (1996) for the 0.95 confidence
default. Artzner et al. (1999) for the coherent risk measure context.
Convention parameters: `method` (`"historical"`, `"parametric"`,
`"cornish_fisher"`), `confidence` (default `0.95`, per Jorion 2006 §5.2).

### 2.10 CVaR / Expected Shortfall

**Historical (default):**
$\text{CVaR}_\alpha = -\mathbb{E}[r \mid r \leq -\text{VaR}_\alpha]$

**Parametric:**
$\text{CVaR}_\alpha = -\bar{r} + \sigma \cdot \phi(z_\alpha) / \alpha$

**Citation:** Rockafellar & Uryasev (2000), "Optimization of Conditional
Value-at-Risk," *Journal of Risk*, 2(3); Acerbi & Tasche (2002). Convention
parameter: `method` (`"historical"`, `"parametric"`), `confidence`
(default `0.95`).

### 2.11 Tail Ratio

$$\text{TR}_\alpha = \frac{\mathbb{E}[r_t \mid r_t \geq q_{1-\alpha}]}
{|\mathbb{E}[r_t \mid r_t \leq q_\alpha]|}$$

Ratio of the conditional mean of returns in the upper $\alpha$-tail to
the absolute conditional mean of returns in the lower $\alpha$-tail.
(Note: `stats.md` §2 #11's "95th percentile / |5th percentile|" is a
plain-language shorthand; the formal definition uses tail conditional
expectations, not raw percentile values.)

**Citation:** Connor, Goldberg & Korajczyk (2010, *Portfolio Risk
Analysis*, Ch. 9). Convention parameter: `tail_cutoff` — default `0.05`.

### 2.12 Common-Sense Ratio

$$\text{CSR} = \text{TR}_\alpha \times
\frac{\sum\max(r_t,0)}{|\sum\min(r_t,0)|}
= \text{TR}_\alpha \times \text{Gain-to-Pain Ratio}$$

**Citation:** Bacon (2008, §7.5).

### 2.13 Hill Tail Index (EVT)

$$\hat{\xi} = \frac{1}{k}\sum_{i=1}^{k}
\ln\frac{X_{(i)}}{X_{(k+1)}}$$

where $X_{(i)}$ are descending order statistics of the upper tail and
$k$ = `tail_fraction` × $n$.

**Citation:** Hill (1975), "A Simple General Approach to Inference About
the Tail of a Distribution," *Annals of Statistics*, 3(5). Convention
parameter: `tail_fraction` — default `0.10`.

### 2.14 GPD Tail Fit

Fit the Generalized Pareto Distribution to exceedances above threshold
$u$ (90th percentile of negative returns):

$$G_{\xi,\beta}(x) = 1 - \left(1 + \xi\frac{x}{\beta}\right)^{\!-1/\xi}$$

**Citation:** Pickands (1975); Hosking & Wallis (1987); Embrechts,
Klüppelberg & Mikosch (1997, *Modelling Extremal Events*, Springer).

### 2.15 Risk of Ruin

$$P_{\text{ruin}} = \Phi\!\left(
\frac{-\bar{r} \cdot T}{\sigma\sqrt{T}}\right)$$

Normal-approximation probability of 100 % loss.

**⚠ Caveat:** Assumes normality — unreliable for fat-tailed returns.
Docstring must cite this limitation and warn that the estimate is a
lower bound for leptokurtic distributions.

**Citation:** Vince (1990, *Portfolio Management Formulas*); standard
in risk-of-ruin literature.

### 2.16 Drawdown Volatility

$$\sigma_{\text{DD}} = \text{std}(d_1, \dots, d_n)$$

where $d_t = (P_t - \max_{\tau \leq t}P_\tau) / \max_{\tau \leq t}P_\tau$
is the drawdown time series.

**Citation:** Bacon (2008, §7.3) discusses drawdown-series statistics.

### 2.17 Drawdown Periods Count

$$K = \text{number of distinct drawdown episodes}$$

An episode begins when $P_t$ falls below the running maximum and ends
when it next returns to the running maximum.

**Citation:** Bacon (2008, §7.2).

### 2.18 Current Drawdown

$$d_{\text{current}} = \frac{P_n - \max_{\tau \leq n}P_\tau}
{\max_{\tau \leq n}P_\tau}$$

Drawdown at the most recent observation.

**Citation:** Bacon (2008, §7.2).

### 2.19 Current Drawdown Duration

Periods elapsed from the most recent running-maximum peak to the current
observation.

**Citation:** Bacon (2008, §7.2).

### 2.20 Drawdown Total Duration

$$T_{\text{DD}}^{\text{total}} = \sum_{k=1}^{K} T_{\text{DD}}^{(k)}$$

Sum of all underwater-period lengths.

**Citation:** Bacon (2008, §7.2).

### 2.21 Pain Index

$$\text{PI} = \frac{1}{n}\sum_{t=1}^{n} d_t, \qquad
d_t = \frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau}$$

Mean of percentage drawdowns over ALL periods, including periods at zero
drawdown. Unlike average drawdown (§2.4), which averages only across
underwater episodes, the Pain Index is smaller in magnitude (less negative)
for strategies that spend time at new highs.

**Citation:** Zephyr Associates; Becker (2006).

### 2.22 Prospect Ratio

$$\text{PR} = \frac{\text{USV}}{\text{DSV}}, \qquad
\text{USV} = \frac{1}{n}\sum_{t=1}^{n}\max(r_t - \text{mar},\,0)^2, \qquad
\text{DSV} = \frac{1}{n}\sum_{t=1}^{n}\min(r_t - \text{mar},\,0)^2$$

Ratio of upside semivariance to downside semivariance. The denominator DSV
is the downside semi-variance (§2.23) — the raw second moment below MAR.
Values > 1 indicate gain dispersion exceeds loss dispersion; < 1 the
reverse. Inf when there is no downside; NaN when there is neither upside
nor downside.

**Citation:** Watanabe (2005). Convention parameter: `mar` — default `0.0`.

### 2.23 Downside Semi-Variance

$$\text{DSV} = \frac{1}{n}\sum_{t=1}^{n}\min(r_t - \tau,\; 0)^2$$

Raw second moment below the minimum acceptable return $\tau$
(default 0). Unlike downside deviation (§2.7), this is not square-rooted,
so it is additive across positions and suits portfolio optimisation.

**Citation:** Markowitz (1959); Sortino & van der Meer (1991).

### 2.24 Modified VaR

$$\text{MVaR} = -(\mu + z_{\text{CF}}\cdot\sigma)$$

Cornish-Fisher VaR, where $z_{\text{CF}}$ adjusts the normal quantile
for sample skewness and excess kurtosis. Same expansion as
`var(method="cornish_fisher")`, registered standalone.

**Citation:** Favre & Galeano (2002); Zangari (1996).

---

## 3. `core.returns` — Risk-Adjusted

Tagged `("risk_adjusted", "returns")`, backend `"vectorized"`.

### 3.1 Sharpe Ratio

$$\text{SR} = \frac{\bar{r}_{\text{excess}}}{\sigma} \cdot \sqrt{P}$$

where $\bar{r}_{\text{excess}} = \bar{r} - r_f$ (risk-free rate, default
0). Annualization via √P is applied pre-division: (r̄_excess · √P) / σ
is equivalent to (r̄_excess · P) / (σ · √P). The `periods_per_year` factor
used for annualization comes from the input container, not from a
per-metric convention.

**Citation:** Sharpe (1966), "Mutual Fund Performance," *Journal of
Business*, 39(1); Sharpe (1994), "The Sharpe Ratio," *JPM*, 21(1).
Convention parameter: `ddof` — `1` (default, sample std) or `0`
(population std).

### 3.2 Sortino Ratio

$$\text{Sortino} = \frac{\bar{r}_{\text{excess}} \cdot P}
{\text{DD} \cdot \sqrt{P}}$$

where DD is downside deviation (§2.7).

**Citation:** Sortino & Price (1994), "Performance Measurement in a
Downside Risk Framework," *Journal of Investing*, 3(3). Convention
parameter: `denominator` — `"full_downside"` (default, divides by
sqrt of mean squared downside over all periods) or `"downside_only"`
(divides by sqrt of mean squared downside over only downside periods).

### 3.3 Calmar Ratio

$$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$

**Citation:** Young (1991), "Calmar Ratio: A Smoother Tool," *Futures*,
20(10).

### 3.4 Omega Ratio

$$\Omega(\tau) = \frac{\sum_{t=1}^{n}\max(r_t - \tau,\, 0)}
{|\sum_{t=1}^{n}\min(r_t - \tau,\, 0)|}$$

**Citation:** Keating & Shadwick (2002), "A Universal Performance
Measure," *JPM*, 6(3). Default threshold $\tau = 0$.

### 3.5 Sterling Ratio

$$\text{Sterling} = \frac{\text{CAGR}}{|\text{ADD}| + k}$$

where $k$ is a floor constant to avoid division by zero for strategies
with very shallow drawdowns.

**Citation:** Industry convention; Bacon (2008, §8.3). Convention
parameter: `floor` — default `0.10` (10 %).

### 3.6 Burke Ratio

$$\text{Burke} = \frac{\text{CAGR}}{\sqrt{\sum_{t=1}^{n} d_t^{2}}}$$

where $d_t = (P_t - \max_{\tau \leq t}P_\tau) / \max_{\tau \leq t}P_\tau$
is the per-period percentage drawdown.

**⚠ Note:** This definition (CAGR divided by root of sum of squared
drawdowns) matches `stats.md` §3 #6. It differs from Bacon (2008, §8.5),
which defines a "Burke ratio" as excess return over the square root of
the mean of squared negative returns — a different numerator and a
different denominator. No single clean canonical source has been
identified for the drawdown-based formulation; flag for project owner
review.

### 3.7 Kappa-3

$$\text{Kappa}_3(\tau) = \frac{\bar{r} - \tau}
{\sqrt[3]{\frac{1}{n}\sum_{t=1}^{n}\max(\tau - r_t,\; 0)^3}}$$

Lower partial moment of order 3.

**Citation:** Kaplan & Knowles (2004), "Kappa: A Generalized Downside
Risk-Adjusted Performance Measure," *JPM*. Convention parameter: `mar` —
default `0.0`.

### 3.8 Martin Ratio (CAGR / Ulcer)

$$\text{Martin} = \frac{\text{CAGR}}{\text{UI}}$$

CAGR divided by Ulcer Index (§2.6). Matches `stats.md` §3 #8 note:
"CAGR / ulcer index."

**Citation:** Martin & McCann (1989); Bacon (2008, §8.6).

### 3.9 Gain-to-Pain Ratio

$$\text{GPR} = \frac{\sum_{t=1}^{n}\max(r_t, 0)}
{|\sum_{t=1}^{n}\min(r_t, 0)|}$$

**Citation:** Bacon (2008, §8.4).

### 3.10 Pain Ratio

$$\text{Pain Ratio} = \frac{\text{CAGR}}{|\text{PI}|}$$

CAGR divided by the absolute value of the Pain Index (§2.21). Higher values
indicate a better return per unit of drawdown pain. Inf when the Pain Index
is zero; NaN when CAGR is undefined.

**Citation:** Zephyr Associates.

### 3.11 Recovery Factor

$$\text{RF} = \frac{\prod_{t=1}^{n}(1 + r_t) - 1}{|\text{MDD}|}$$

Total cumulative return divided by absolute max drawdown (§2.1). Uses TOTAL
return, not CAGR (which is the Calmar ratio, §3.3). Inf when max drawdown is
zero.

**Citation:** Industry convention.

### 3.12 K-Ratio

$$y_t = \sum_{\tau=1}^{t}\ln(1 + r_\tau), \qquad
y_t = \alpha + \beta\,t + \varepsilon_t, \qquad
K = \frac{\beta}{\text{SE}(\beta)}, \qquad
\text{SE}(\beta) = \sqrt{\frac{\text{MSE}}{\sum_t (t - \bar{t})^2}}$$

Slope of the log-VAMI regression divided by its standard error; measures the
consistency of equity-curve growth. Higher values indicate a smoother, more
linear curve. Requires at least 3 periods.

**Citation:** Kestner (1996), revised 2003.

### 3.13 Serenity Ratio

$$\text{Serenity} = \frac{R_p - R_f}{\sigma_p \cdot \text{UI}}$$

where $R_p$ is the annualized arithmetic mean return, $R_f$ the annualized
risk-free rate, $\sigma_p$ the annualized volatility (§1.2), and UI the
Ulcer Index (§2.6). Both volatility and drawdown risk must be low for a high
score. NaN when volatility or Ulcer Index is zero.

**Citation:** Industry metric; used by PortfolioMetrics.

### 3.14 UPI (Ulcer Performance Index)

$$\text{UPI} = \frac{R_p - R_f}{\text{UI}}$$

Same numerator as the Serenity ratio (§3.13) but divided by the Ulcer Index
alone, not the product with volatility. Differs from the Martin ratio
(§3.8), which uses CAGR in the numerator. NaN when the Ulcer Index is zero.

**Citation:** Martin & McCann (1989).

### 3.15 Modified Sharpe Ratio

$$\text{MSR} = \frac{R_p - R_f}{\text{MVaR}}$$

Excess return divided by Modified VaR (§2.24), which uses the Cornish-Fisher
expansion to adjust for skewness and excess kurtosis rather than assuming
normality. NaN when Modified VaR is zero.

**Citation:** Gregoriou & Gueyie (2003). Convention parameter: `confidence`
— default `0.95`.

### 3.16 Upside Potential Ratio

$$\text{UPR} = \frac{\frac{1}{n}\sum_{t=1}^{n}\max(r_t - \text{mar},\,0)}
{\sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(r_t - \text{mar},\,0)^2}}$$

Sortino variant that replaces the mean excess return in the numerator with
upside potential (the mean of only positive excess returns). The denominator
is downside deviation (§2.7). Inf when there is no downside; NaN when there
is neither upside nor downside.

**Citation:** Sortino, van der Meer & Plantinga (1999). Convention
parameter: `mar` — default `0.0`.

### 3.17 Risk Return Ratio

$$\text{RRR} = \frac{\bar{r} \cdot P}{|\text{MDD}|}$$

Annualized arithmetic return divided by absolute max drawdown (§2.1).
Simpler than the Calmar ratio (§3.3), which uses CAGR. Inf when max drawdown
is zero.

**Citation:** Industry convention.

### 3.18 Roy's Safety-First Ratio

$$\text{RSF} = \frac{\bar{R}_p - \text{MAR}}{\sigma_p}$$

Annualized mean excess return over the minimum acceptable return (MAR,
default 0), divided by annualized volatility. Ranks portfolios by the
probability of falling short of MAR under a normality assumption.

**Citation:** Roy (1952), "Safety First and the Holding of Assets,"
*Econometrica*, 20(3).

### 3.19 Autocorrelation Penalty (Lo 2002)

$$\rho_1 = \text{Corr}(r_{t}, r_{t-1}),\qquad
\text{penalty} = \sqrt{1 + 2\sum_{k=1}^{n-1}\frac{n-k}{n}\,|\rho_1|^k}$$

The penalty factor is ≥ 1 and shrinks the Sharpe/Sortino ratios to account
for positive serial dependence. A constant series yields a penalty of 1.0.

**Citation:** Lo (2002), "The Statistics of Sharpe Ratios," *Financial
Analysts Journal*, 58(4).

### 3.20 Smart Sharpe Ratio

$$\text{smart\_sharpe} = \frac{\text{Sharpe}}{\text{autocorrelation penalty}}$$

Sharpe ratio (§3.1) divided by the Lo autocorrelation penalty (§3.19).

**Citation:** Lo (2002), *Financial Analysts Journal*, 58(4).

### 3.21 Smart Sortino Ratio

$$\text{smart\_sortino} = \frac{\text{Sortino}}{\text{autocorrelation penalty}}$$

Sortino ratio (§3.2) divided by the Lo autocorrelation penalty (§3.19).

**Citation:** Lo (2002), *Financial Analysts Journal*, 58(4).

### 3.22 Adjusted Sortino Ratio

$$\text{adjusted\_sortino} = \frac{\text{Sortino}}{\sqrt{2}}$$

The Sortino ratio (§3.2) divided by $\sqrt{2}$, so a symmetric return
distribution gives an adjusted Sortino comparable to the Sharpe ratio.

**Citation:** Schwager (2012), *Hedge Fund Market Wizards*.

### 3.23 Risk-Adjusted Return (RAR)

$$\text{RAR} = \frac{\text{CAGR}(r - r_f)}{\text{exposure}}$$

CAGR of the excess returns divided by exposure time (§1.22). A higher value
means more growth per unit of time actually invested. Maps to QuantStats
`rar`.

**Citation:** QuantStats-compatible convention; no independent academic
source identified.

---

## 4. `core.returns` — Inference

Tagged `("inference", "returns")`, backend `"resampling"`.

### 4.1 Jarque-Bera Statistic

$$\text{JB} = \frac{n}{6}\!\left(\gamma_1^2 +
\frac{\gamma_2^2}{4}\right)$$

where $\gamma_1$ is sample skewness (§1.6) and $\gamma_2$ is sample
excess kurtosis (§1.7). Under the null of normality, \(\text{JB} \sim
\chi^2(2)\). (Equivalently: \(\text{JB} = \frac{n}{6}(S^2 +
\frac{(K-3)^2}{4})\) where $S$ is skewness and $K$ is raw kurtosis;
the form above uses excess kurtosis directly.)

**Citation:** Jarque & Bera (1987), "A Test for Normality of Observations
and Regression Residuals," *International Statistical Review*, 55(2).

### 4.2 Probabilistic Sharpe Ratio (PSR)

$$\text{PSR}(SR^*) = \Phi\!\left(
\frac{(\hat{SR} - SR^*)\sqrt{T-1}}
{\sqrt{1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}\right)$$

Probability that true SR exceeds benchmark $SR^*$, adjusted for skewness
and kurtosis.

**Citation:** Bailey & López de Prado (2012), "The Sharpe Ratio Efficient
Frontier," *Journal of Risk*, 15(2); de Prado (2018, *Advances in
Financial Machine Learning*, Ch. 14). Convention parameter:
`sharpe_benchmark` — default `0.0`.

### 4.3 Deflated Sharpe Ratio (DSR)

$$\text{DSR} = \Phi\!\left(
\frac{(\hat{SR} - SR^*)\sqrt{T-1}}
{\sqrt{1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}
\cdot \left(1 - \frac{1}{M}\sum_{m=1}^{M}
\mathbf{1}_{[\text{SR}_m^{(b)} < \hat{SR}]}\right)\!\right)$$

PSR deflated by the probability of multiple-testing false discovery among
$M$ trials.

**Citation:** Bailey & López de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,"
*JPM*, 40(5).

### 4.4 Lo's Autocorrelation-Adjusted Sharpe SE

$$\text{SE}_{\text{IID}}(\hat{SR}) = \sqrt{\frac{1}{T}
\!\left(1 + \frac{1}{2}\hat{SR}^2 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-3}{4}\hat{SR}^2\right)}$$

$$\text{SE}_{\text{adj}} = \text{SE}_{\text{IID}} \cdot
\sqrt{\frac{1+\rho_1}{1-\rho_1}}$$

**Citation:** Lo (2002), "The Statistics of Sharpe Ratios," *Financial
Analysts Journal*, 58(4).

### 4.5 Sharpe Ratio CI — Analytic

$$\hat{SR} \pm z_{1-\alpha/2} \cdot \text{SE}(\hat{SR})$$

using Lo's SE (IID or adjusted per §4.4).

**Citation:** Lo (2002).

### 4.6 Sharpe Ratio CI — Bootstrap

Non-parametric block bootstrap (BCa method): resample overlapping blocks,
recompute SR, form CI from the bootstrap distribution.

**Citation:** Efron & Tibshirani (1994, *An Introduction to the
Bootstrap*); Ledoit & Wolf (2008), "Robust Performance Hypothesis Testing
with the Sharpe Ratio," *J. Empirical Finance*, 15(5). Block-length
selection per Politis & White (2004). Default: 5,000 replications,
automatic block length, 95 % CI.

### 4.7 Minimum Track Record Length

$$T_{\min} = 1 + \left(\frac{z_{1-\alpha}}{\hat{SR} - SR^*}\right)^{\!2}
\cdot \left(1 - \hat{\gamma}_3\hat{SR} +
\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2\right)$$

where $z_{1-\alpha} = \Phi^{-1}(1-\alpha)$ is the critical value and
$\hat{\gamma}_3, \hat{\gamma}_4$ are sample skewness and raw kurtosis
respectively.

**Citation:** Bailey & López de Prado (2012); de Prado (2018, Ch. 14).
Default $\alpha = 0.05$.

### 4.8 Generic Block-Bootstrap CI Wrapper

Moving-block bootstrap: for any registered `returns`-tier metric function
$f$, resample blocks of length $l$ with replacement, recompute $f$ on
each resample, return empirical CI.

**Citation:** Efron & Tibshirani (1994); Künsch (1989), "The Jackknife
and the Bootstrap for General Stationary Observations," *Annals of
Statistics*, 17(3). Politis & White (2004) for block-length selection.
Default: 5,000 reps, automatic block length, 95 % equal-tailed CI.

### 4.9 Bias Ratio

$$\text{Bias Ratio} = \frac{\sum_{t}\mathbf{1}_{[\,|r_t| < b\,\sigma\,]}}
{\max\!\left(\sum_{t}\mathbf{1}_{[\,|r_t| \ge b\,\sigma\,]},\;1\right)}$$

Count of returns within a narrow band around zero (width $b$ standard
deviations, default $b = 1.0$, with $\sigma$ the sample standard deviation
ddof = 1) divided by the count outside the band (floored at 1). Detects
smoothed or manipulated returns: a high value suggests clustering around
zero. Requires at least 2 periods.

**Citation:** Abdulali (2006), "Detecting Smoothed Returns." Convention
parameter: `bandwidth` — default `1.0`.

### 4.10 Skewness-Adjusted Sharpe Ratio (ASR)

$$\text{ASR} = \text{SR}\!\left(1 + \frac{\gamma_3}{6}\text{SR} -
\frac{\gamma_4}{24}\text{SR}^2\right)$$

Adjusts the (annualized) Sharpe ratio (§3.1) for the sample skewness
$\gamma_3$ (§1.6) and sample excess kurtosis $\gamma_4$ (§1.7). For normal
returns ASR ≈ SR; positive skewness raises ASR, excess kurtosis lowers it
(penalising fat tails). Requires at least 4 observations.

**Citation:** Pezier & White (2008).

### 4.11 Probabilistic Sortino Ratio

$$\text{PSortino} = \Phi\!\left(\frac{(\widehat{So} - So^*) \sqrt{n-1}}
{\sqrt{1 - \hat{\gamma}_3 \widehat{So} +
\frac{\hat{\gamma}_4 - 1}{4} \widehat{So}^2}}\right)$$

The probabilistic ratio of Bailey & López de Prado applied to the period
(non-annualized) Sortino base instead of the Sharpe base. $\widehat{So}$ is
the period Sortino ratio, $So^*$ the benchmark Sortino at period frequency
(default 0), and $\Phi$ the standard normal CDF.

**Citation:** Bailey & López de Prado (2012), "The Sharpe Ratio Efficient
Frontier," *JPM*, 38(5); QuantStats-compatible extension to the Sortino
base.

### 4.12 Probabilistic Adjusted Sortino Ratio

$$\text{PAdjSortino} = \Phi\!\left(\frac{(\widehat{So}/\sqrt{2} - So^*) \sqrt{n-1}}
{\sqrt{1 - \hat{\gamma}_3 \widehat{So}/\sqrt{2} +
\frac{\hat{\gamma}_4 - 1}{4} (\widehat{So}/\sqrt{2})^2}}\right)$$

The same construction as §4.11, with the adjusted Sortino base
($\widehat{So}/\sqrt{2}$) in place of the Sortino base.

**Citation:** Bailey & López de Prado (2012); Schwager (2012) adjusted
Sortino base; QuantStats-compatible extension.

### 4.13 Monte Carlo Distribution

Non-parametric with-replacement bootstrap (Efron). Resample the historical
returns $n$ periods at a time, `sims` times (default 1000), compute the
chosen terminal statistic on each path, and return the summary

$$[\min,\; p_{05},\; \text{median},\; \text{mean},\; p_{95},\; \max,\; \sigma]$$

`target` selects the terminal statistic: `"equity"` (total return),
`"sharpe"` (annualized), `"max_drawdown"`, or `"cagr"` (annualized).
Sampling is **with replacement** — a permutation would leave the mean,
standard deviation, and terminal equity invariant, yielding degenerate
distributions for the Sharpe, CAGR, and equity targets.

**Citation:** Efron & Tibshirani (1994, *An Introduction to the
Bootstrap*).

### 4.14 Monte Carlo Probabilities

Bootstrap the same paths as §4.13 and report

$$\left[p_{\text{bust}},\; p_{\text{goal}}\right] =
\left[\Pr(\text{MDD} \le \text{bust}),\;
\Pr(R_{\text{terminal}} \ge \text{goal})\right]$$

where `bust` is a (negative) drawdown threshold and `goal` a terminal-return
threshold. A threshold left as `None` yields NaN for its probability.

**Citation:** Efron & Tibshirani (1994, *An Introduction to the
Bootstrap*).

---

## 5. `core.returns` — Rolling / Regime Wrappers

Generic functions applied via the registry to any `returns`-tier metric.

### 5.1 `rolling(metric_name, window)`

Apply the named metric over a rolling window of `window` periods,
producing a time series of that metric.

**Citation:** No single canonical source — rolling-window analysis is a
standard technique in time-series econometrics. For a formal treatment,
see Zivot & Wang (2006, *Modeling Financial Time Series with S-PLUS*,
Ch. 3).

### 5.2 `by_regime(metric_name, regime_labels)`

Group returns by regime label (same-length array), compute the named
metric separately per regime.

**Citation:** Ang & Bekaert (2004), "How Regimes Affect Asset
Allocation," *Financial Analysts Journal*, 60(2); standard
regime-switching analysis.

---

## 6. `core.exposure`

Tagged `requires="exposure"`. Category tags vary.

### 6.1 Gross Exposure

$$\text{GE}_t = \sum_{i=1}^{N} \left|\frac{\text{position\_value}_{i,t}}
{\text{portfolio\_value}_t}\right|$$

Report: current, max, average.

**Citation:** Ang (2014, *Asset Management: A Systematic Approach to
Factor Investing*, Oxford University Press, Ch. 2).

### 6.2 Net Exposure

$$\text{NE}_t = \sum_{i=1}^{N} \frac{\text{position\_value}_{i,t}}
{\text{portfolio\_value}_t}$$

Report: current, max, min, average, range.

**Citation:** Ang (2014, Ch. 2).

### 6.3 Leverage

$$\text{Leverage}_t = \frac{\text{gross\_exposure}_t}{\text{equity}_t}$$

Ratio of gross exposure to equity.

**Citation:** Ang (2014, Ch. 2).

### 6.4 Long Exposure %

$$\text{LE}\%_t = \sum_{i} w_{i,t} \cdot \mathbf{1}_{[w_{i,t} > 0]}$$

Fraction of gross exposure allocated long.

**Citation:** CFA Institute, *GIPS Standards* (2020); also Bacon (2008,
§11.3).

### 6.5 Short Exposure %

$$\text{SE}\%_t = \sum_{i} |w_{i,t}| \cdot \mathbf{1}_{[w_{i,t} < 0]}$$

Fraction of gross exposure allocated short.

**Citation:** As §6.4.

### 6.6 Long-Book Contribution to Return

$$R_t^{\text{long}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
\mathbf{1}_{[w_{i,t-1} > 0]}$$

**Citation:** Bacon (2008, §11.5), "Performance Attribution."

### 6.7 Short-Book Contribution to Return

$$R_t^{\text{short}} = \sum_{i} w_{i,t-1} \cdot r_{i,t} \cdot
\mathbf{1}_{[w_{i,t-1} < 0]}$$

**Citation:** As §6.6.

### 6.8 Long Beta

$$\beta_{\text{long}} = \frac{\text{Cov}(R^{\text{long}}, R_m)}
{\text{Var}(R_m)}$$

**Requires:** optional benchmark field on `ExposureInput`. If benchmark is
not provided, raises `ValueError`.

**Citation:** Asness, Frazzini & Pedersen (2014), "Low-Risk Investing
Without Industry Bets," *Financial Analysts Journal*, 70(4).

### 6.9 Short Beta

$$\beta_{\text{short}} = \frac{\text{Cov}(R^{\text{short}}, R_m)}
{\text{Var}(R_m)}$$

**Requires:** optional benchmark field on `ExposureInput`. Same failure
mode as §6.8.

**Citation:** As §6.8.

### 6.10 Position Concentration (HHI)

$$\text{HHI}_t = \sum_{i=1}^{N}
\left(\frac{w_{i,t}}{\sum_j |w_{j,t}|}\right)^{\!2}$$

Herfindahl-Hirschman Index on normalized position weights.

**Citation:** Hirschman (1964), "The Paternity of an Index," *American
Economic Review*, 54(5); SEC regulation for concentration reporting.

### 6.11 Effective N Positions

$$N_{\text{eff},t} = \frac{1}{\text{HHI}_t}$$

Reciprocal HHI.

**Citation:** Adelman (1969), "Comment on the 'H' Concentration Measure
as a Numbers-Equivalent," *Review of Economics and Statistics*, 51(1).

### 6.12 Turnover

$$\text{TO}_t = \frac{1}{2}\sum_{i=1}^{N}
|\Delta w_{i,t}|$$

Annualized: mean period turnover × $P$.

**Citation:** Morningstar (2020), *Morningstar Portfolio Turnover
Methodology*; also Bacon (2008, §11.6).

### 6.13 Average Holding Weight per Position

$$\bar{w}_t = \frac{1}{N_t}\sum_{i=1}^{N_t} |w_{i,t}|$$

**Citation:** Bacon (2008, §11.2).

### 6.14 Position Coverage %

$$\text{Coverage} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[\exists i: w_{i,t} \neq 0]}$$

Fraction of periods with at least one non-zero position.

**Citation:** Bacon (2008, §11.2).

### 6.15 Long Position Coverage %

$$\text{Coverage}_{\text{long}} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[\exists i: w_{i,t} > 0]}$$

Fraction of periods with at least one long (strictly positive) position.

**Citation:** As §6.14.

### 6.16 Short Position Coverage %

$$\text{Coverage}_{\text{short}} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[\exists i: w_{i,t} < 0]}$$

Fraction of periods with at least one short (strictly negative) position.

**Citation:** As §6.14.

### 6.17 Exposure Volatility

$$\sigma_{\text{GE}} = \text{std}(\text{GE}_1, \dots, \text{GE}_n)$$

Standard deviation of the gross exposure time series.

**Citation:** Bacon (2008, §11.3).

### 6.18 Net Exposure Volatility

$$\sigma_{\text{NE}} = \text{std}(\text{NE}_1, \dots, \text{NE}_n)$$

**Citation:** As §6.17.

### 6.19 Exposure Coefficient of Variation

$$\text{CV}_{\text{exp}} =
\frac{\sigma_{\text{GE}}}{|\bar{\text{GE}}|}$$

**Citation:** Pearson (1896). For financial application: standard
descriptive statistic; no single finance-specific source.

### 6.20 Avg Exposure Utilization

$$\text{Utilization} = \frac{\bar{\text{GE}}}{\max_t \text{GE}_t}$$

Mean gross exposure as a fraction of maximum.

**Citation:** Bacon (2008, §11.3).

### 6.21 Exposure Directional Bias

$$\text{Bias} = \frac{|\bar{\text{NE}}|}{\bar{\text{GE}}}$$

Absolute mean net exposure relative to mean gross exposure.

**Citation:** Bacon (2008, §11.3); Ang (2014, Ch. 2).

### 6.22 Exposure Percentiles

Percentiles at {25, 50, 75, 90, 95} of the gross exposure time series.

**Citation:** Hyndman & Fan (1996); standard order statistics.

### 6.23 Period Counts

Breakdown: total periods, periods with any position, long-only periods,
short-only periods, idle (flat) periods.

**Citation:** Bacon (2008, §11.2).

---

## 7. `core.trades`

Tagged `requires="trades"`. Category tags vary.

All metrics in this section are standard trade-analysis statistics. The
primary source is Schwager (1995, *Schwager on Futures: Technical
Analysis*, Wiley). Additional sources cited per metric where applicable.

### 7.1 Total Trades

$$N = \text{number of round-trip trades}$$

**Citation:** Schwager (1995, Ch. 38).

### 7.2 Win Rate (Overall)

$$\text{WR} = \frac{N_{\text{win}}}{N}$$

Fraction of trades with positive P&L.

**Citation:** Schwager (1995, Ch. 38).

### 7.3 Win Rate (Long-Only)

Same formula, restricted to trades opened long. A trade is "long" if
`side` in the trade log equals `"long"` (or equivalent convention per
the `TradeInput` schema).

### 7.4 Win Rate (Short-Only)

Same formula, restricted to trades opened short.

### 7.5 Average Win

$$\bar{W} = \frac{1}{N_{\text{win}}}\sum_{j:\text{win}}\text{PnL}_j$$

**Citation:** Schwager (1995, Ch. 38).

### 7.6 Average Loss

$$\bar{L} = \frac{1}{N_{\text{loss}}}\sum_{j:\text{loss}}\text{PnL}_j$$

Reported as absolute value for display; sign preserved in the
`MetricResult.value` field.

**Citation:** Schwager (1995, Ch. 38).

### 7.7 Win/Loss Ratio

$$\text{WLR} = \frac{N_{\text{win}}}{N_{\text{loss}}}$$

**Citation:** Schwager (1995, Ch. 38).

### 7.8 Profit Factor

$$\text{PF} = \frac{\sum_{j}\max(\text{PnL}_j, 0)}
{|\sum_{j}\min(\text{PnL}_j, 0)|}$$

**Citation:** Schwager (1995, Ch. 38).

### 7.9 Expectancy per Trade

$$\mathbb{E}[\text{PnL}] = \text{WR} \cdot \bar{W} +
(1 - \text{WR}) \cdot \bar{L}$$

with $\bar{L}$ as a negative number.

**Citation:** Tharp (1998, *Trade Your Way to Financial Freedom*,
McGraw-Hill, Ch. 5).

### 7.10 Average Holding Period

$$\bar{H} = \frac{1}{N}\sum_{j=1}^{N}(t_{j,\text{exit}} -
t_{j,\text{entry}})$$

**Citation:** Schwager (1995, Ch. 38).

### 7.11 Holding Period Distribution

Median, 25th, 75th percentile, min, max of trade holding periods.

**Citation:** Standard descriptive statistic; see Hyndman & Fan (1996)
for quantile interpolation method.

### 7.12 Max Consecutive Wins

Longest run of consecutive winning trades.

**Citation:** Schwager (1995, Ch. 38); Tharp (1998, Ch. 5).

### 7.13 Max Consecutive Losses

Longest run of consecutive losing trades.

**Citation:** As §7.12.

### 7.14 Round-Trip P&L Distribution

Summary statistics of per-trade P&L: mean, median, std, skewness,
5th/95th percentiles.

**Citation:** Schwager (1995, Ch. 38); Tharp (1998, Ch. 7, "Expectunity").

### 7.15 Implementation Shortfall

$$\text{IS}_j = \text{side}_j \cdot
\frac{P_{\text{fill},j} - P_{\text{decision},j}}
{P_{\text{decision},j}}$$

**Required optional fields on the trade log:**
`fill_price`, `decision_price`, `side` (sign: +1 for buy, −1 for sell).

If any of these fields is absent from the trade log, raises `ValueError`
with a message listing which fields are missing.

**Citation:** Perold (1988), "The Implementation Shortfall: Paper versus
Reality," *JPM*, 14(3).

### 7.16 Best Trade

$$\max_j \text{PnL}_j$$

**Citation:** Schwager (1995, Ch. 38).

### 7.17 Worst Trade

$$\min_j \text{PnL}_j$$

**Citation:** As §7.16.

### 7.18 Avg Winning Trade Duration

$$\bar{H}_{\text{win}} = \frac{1}{N_{\text{win}}}
\sum_{j:\text{win}}(t_{j,\text{exit}} - t_{j,\text{entry}})$$

**Citation:** Schwager (1995, Ch. 38).

### 7.19 Avg Losing Trade Duration

$$\bar{H}_{\text{loss}} = \frac{1}{N_{\text{loss}}}
\sum_{j:\text{loss}}(t_{j,\text{exit}} - t_{j,\text{entry}})$$

**Citation:** As §7.18.

### 7.20 Payoff Ratio

$$\text{Payoff} = \frac{\bar{W}}{|\bar{L}|}$$

Average win divided by absolute average loss.

**Citation:** Schwager (1995, Ch. 38).

### 7.21 CPC Ratio

$$\text{CPC} = \text{PF} \times \text{Payoff} \times \text{WR}$$

Young's CPC Index: profit factor × payoff ratio × win rate.

**Citation:** Young (1991), CPC Index; see also Schwager (1995, Ch. 38).

### 7.22 SQN (System Quality Number)

$$\text{SQN} = \frac{\bar{r}_{\text{trade}}}
{\sigma_{\text{trade}}} \cdot \sqrt{N}$$

Mean trade return divided by trade return std, scaled by √(number of
trades).

**Citation:** Tharp (1998, Ch. 5).

### 7.23 Trade Duration Std

$$\sigma_H = \text{std}(H_1, \dots, H_N)$$

**Citation:** Standard statistic. Fisher (1925).

### 7.24 Trade Return Std

$$\sigma_{\text{trade}} = \text{std}(\text{PnL}_1, \dots, \text{PnL}_N)$$

**Citation:** Standard statistic. Fisher (1925).

### 7.25 Geometric Mean Return per Trade

$$\bar{r}_{g,\text{trade}} = \exp\!\left(
\frac{1}{N}\sum_{j=1}^{N}\ln(1 + \text{PnL}_j)\right) - 1$$

**Citation:** Standard geometric mean; see Campbell, Lo & MacKinlay
(1997, §1.4) for context on compounding.

### 7.26 Outlier Win Ratio

Fraction of winning trades with P&L exceeding \(Q_3 + 1.5 \times
\text{IQR}\) of the winning-trade P&L distribution.

**Citation:** Tukey (1977) for the outlier criterion.

### 7.27 Outlier Loss Ratio

Fraction of losing trades with P&L below $Q_1 - 1.5 \times \text{IQR}$
of the losing-trade P&L distribution.

**Citation:** As §7.26.

### 7.28 MFE (Maximum Favorable Excursion)

For each trade, the maximum dollar gain relative to entry price observed
during the trade's lifetime.

**Requires:** optional intra-trade price path field (`intratrade_prices`)
on `TradeInput`. If this field is absent, raises `ValueError` with a
message: `"MFE requires intratrade_prices; provide an intra-trade price
path in TradeInput"`.

**Citation:** Standard trade analytics; see Sweeney (1988), "The
Maximum Favorable Excursion Methodology."

### 7.29 MAE (Maximum Adverse Excursion)

For each trade, the maximum dollar loss relative to entry price observed
during the trade's lifetime.

**Requires:** same `intratrade_prices` field as MFE (§7.28). Same
failure mode: raises `ValueError` if the field is absent.

**Citation:** As §7.28.

### 7.30 Kelly Criterion

$$f^* = W - \frac{1-W}{\bar{W}/|\bar{L}|}$$

Estimated optimal bet fraction. Assumes independent, identically
distributed trade returns.

**Citation:** Kelly (1956), "A New Interpretation of Information Rate,"
*Bell System Technical Journal*, 35(4); Thorp (1997), "The Kelly
Criterion in Blackjack, Sports Betting, and the Stock Market."

### 7.31 Long/Short Trade Count

$$N_{\text{long}} = \sum_{j=1}^{N} \mathbf{1}_{[\text{side}_j = \text{long}]},
\quad N_{\text{short}} = \sum_{j=1}^{N} \mathbf{1}_{[\text{side}_j = \text{short}]}$$

Number of trades opened long (short). The `side` field in the trade log
determines classification.

### 7.32 Long/Short Trade %

$$p_{\text{long}} = \frac{N_{\text{long}}}{N},\quad
p_{\text{short}} = \frac{N_{\text{short}}}{N}$$

Fraction of all trades opened long (short).

### 7.33 Long/Short Winning/Losing Trades

$$\begin{aligned}
N_{\text{long,win}} &= \sum_{j: \text{side}_j = \text{long}} \mathbf{1}_{[\text{PnL}_j > 0]} \\
N_{\text{long,loss}} &= \sum_{j: \text{side}_j = \text{long}} \mathbf{1}_{[\text{PnL}_j < 0]} \\
N_{\text{short,win}} &= \sum_{j: \text{side}_j = \text{short}} \mathbf{1}_{[\text{PnL}_j > 0]} \\
N_{\text{short,loss}} &= \sum_{j: \text{side}_j = \text{short}} \mathbf{1}_{[\text{PnL}_j < 0]}
\end{aligned}$$

### 7.34 Long/Short Avg Duration

$$\bar{H}_{\text{long}} = \frac{1}{N_{\text{long}}}
\sum_{j: \text{side}_j = \text{long}} H_j,\quad
\bar{H}_{\text{short}} = \frac{1}{N_{\text{short}}}
\sum_{j: \text{side}_j = \text{short}} H_j$$

### 7.35 Long/Short Total PnL %

$$\text{PnL}_{\text{long,total}} = \sum_{j: \text{side}_j = \text{long}} \text{PnL}_j,\quad
\text{PnL}_{\text{short,total}} = \sum_{j: \text{side}_j = \text{short}} \text{PnL}_j$$

### 7.36 Long/Short Avg PnL %

$$\overline{\text{PnL}}_{\text{long}} = \frac{\text{PnL}_{\text{long,total}}}{N_{\text{long}}},\quad
\overline{\text{PnL}}_{\text{short}} = \frac{\text{PnL}_{\text{short,total}}}{N_{\text{short}}}$$

### 7.37 Long/Short Best/Worst Trade %

$$\begin{aligned}
\text{Best}_{\text{long}} &= \max_{j: \text{side}_j = \text{long}} \text{PnL}_j,
\quad \text{Worst}_{\text{long}} = \min_{j: \text{side}_j = \text{long}} \text{PnL}_j \\
\text{Best}_{\text{short}} &= \max_{j: \text{side}_j = \text{short}} \text{PnL}_j,
\quad \text{Worst}_{\text{short}} = \min_{j: \text{side}_j = \text{short}} \text{PnL}_j
\end{aligned}$$

---

## 8. `core.benchmark`

Tagged `requires="benchmark"`. Category tags vary.

### 8.1 Alpha (Jensen's)

$$\alpha_{\text{ann}} = \text{CAGR} - (r_f + \beta \cdot
(\text{CAGR}_m - r_f))$$

**Citation:** Jensen (1968), "The Performance of Mutual Funds in the
Period 1945–1964," *Journal of Finance*, 23(2).

### 8.2 Beta

$$\beta = \frac{\text{Cov}(r, r_m)}{\text{Var}(r_m)}$$

**Citation:** Sharpe (1964), "Capital Asset Prices: A Theory of Market
Equilibrium Under Conditions of Risk," *Journal of Finance*, 19(3).
Convention parameter: `variant` — `"least_squares"` (default, OLS).

### 8.3 R²

$$R^2 = 1 - \frac{\text{Var}(r - \hat{r})}{\text{Var}(r)}$$

where $\hat{r} = \alpha + \beta r_m$.

**Citation:** Greene (2018, *Econometric Analysis*, 8th ed., §3.5); the
coefficient of determination in the OLS context.

### 8.4 Tracking Error

$$\text{TE}_{\text{ann}} = \sigma(r - r_m) \cdot \sqrt{P}$$

**Citation:** Roll (1992), "A Mean/Variance Analysis of Tracking Error,"
*JPM*, 18(4).

### 8.5 Information Ratio

$$\text{IR}_{\text{ann}} = \frac{(\bar{r} - \bar{r}_m) \cdot P}
{\sigma(r - r_m) \cdot \sqrt{P}}$$

**Citation:** Goodwin (1998), "The Information Ratio," *Financial
Analysts Journal*, 54(4).

### 8.6 Up-Capture Ratio

$$\text{UC} = \frac{\text{mean}_{t: r_{m,t} > 0}(r_t)}
{\text{mean}_{t: r_{m,t} > 0}(r_{m,t})}$$

Ratio of mean strategy return to mean benchmark return in up-market
periods.

**Citation:** Morningstar (2020), *Morningstar Performance Reporting
Methodology*; Bacon (2008, §9.4).

### 8.7 Down-Capture Ratio

$$\text{DC} = \frac{\text{mean}_{t: r_{m,t} < 0}(r_t)}
{\text{mean}_{t: r_{m,t} < 0}(r_{m,t})}$$

**Citation:** As §8.6.

### 8.8 Up/Down Capture Ratio

$$\text{UDR} = \frac{\text{UC}}{|\text{DC}|}$$

**Citation:** Bacon (2008, §9.4).

### 8.9 Correlation

$$\rho = \frac{\text{Cov}(r, r_m)}{\sigma(r)\cdot\sigma(r_m)}$$

Pearson correlation with benchmark.

**Citation:** Pearson (1895); standard statistic.

### 8.10 Active Return

$$\bar{r}_{\text{active, ann}} = (\bar{r} - \bar{r}_m) \cdot P$$

**Citation:** Bacon (2008, §9.2); CFA Institute, *Performance
Attribution*.

### 8.11 Batting Average vs Benchmark

$$\text{BA} = \frac{1}{n}\sum_{t=1}^{n}
\mathbf{1}_{[r_t > r_{m,t}]}$$

**Citation:** Bacon (2008, §9.5).

### 8.12 Treynor Ratio

$$\text{Treynor} = \frac{\bar{r}_{\text{excess}} \cdot P}{\beta}$$

**Citation:** Treynor (1965), "How to Rate Management of Investment
Funds," *Harvard Business Review*, 43(1).

### 8.13 Outperformance

$$R_{\text{out}} = R_{\text{cum}} - R_{m,\text{cum}}$$

Total cumulative return difference vs. benchmark.

**Citation:** Bacon (2008, §9.2).

### 8.14 Outperformance Ratio

$$\text{OR} = \frac{1 + R_{\text{cum}}}{1 + R_{m,\text{cum}}}$$

**Citation:** Bacon (2008, §9.2).

### 8.15 Underperforming Periods / %

Count and fraction of periods where $r_t < r_{m,t}$.

**Citation:** Bacon (2008, §9.5).

### 8.16 Max Outperformance

Maximum value of the cumulative active return series
$\sum_{\tau=1}^{t}(r_\tau - r_{m,\tau})$ over the full period.

**Citation:** Bacon (2008, §9.3).

### 8.17 Max Underperformance

Maximum absolute value of cumulative active return below zero (i.e., the
deepest cumulative underperformance vs. benchmark).

**Citation:** As §8.16.

### 8.18 Benchmark Volatility

$$\sigma_{m,\text{ann}} = \sigma(r_m) \cdot \sqrt{P}$$

**Citation:** Standard risk metric; CFA Institute, *Quantitative
Methods*.

### 8.19 Information Coefficient (Rank IC)

$$\text{IC} = \rho_{\text{Spearman}}(\text{rank}(r_p), \text{rank}(r_m))$$

Spearman rank correlation between strategy and benchmark returns.
Robust to outliers, unlike the Pearson correlation (§8.9).

**Citation:** Spearman (1904); standard rank-based information
coefficient.

### 8.20 Directional Consistency

$$\text{DC} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}_{[\text{sign}(r_{p,t}) = \text{sign}(r_{m,t})]}$$

Fraction of periods where strategy and benchmark returns share the same
sign. Ignores magnitude and captures directional agreement only.

**Citation:** Standard statistic; fraction of periods where strategy and
benchmark returns share the same sign.

---

## 9. `core.compare`

Tagged `requires="compare"`, `category=("relative", "compare")`.

### 9.1 Correlation Matrix

$$\Sigma_{ij} = \text{Corr}(r^{(i)}, r^{(j)})$$
for $i,j = 1,\dots,K$.

**Citation:** Pearson (1895); standard multivariate statistic. See
Johnson & Wichern (2007, *Applied Multivariate Statistical Analysis*,
6th ed., §2.5).

### 9.2 Diversification Ratio

$$\text{DR} = \frac{\sum_{i=1}^{K} w_i\sigma_i}{\sigma_p}$$

with equal-weight (default) or user-specified weights.

**Citation:** Choueifaty & Coignard (2008), "Toward Maximum
Diversification," *JPM*, 35(1).

### 9.3 Pairwise Sharpe-Difference Test (Jobson-Korkie, Memmel)

$$z = \frac{\hat{SR}_1 - \hat{SR}_2}{\sqrt{\hat{\sigma}^2}}
\sim \mathcal{N}(0,1)$$

$$\hat{\sigma}^2 = \frac{1}{T}\!\left[2 - 2\rho_{12} +
\frac{1}{2}(\hat{SR}_1^2 + \hat{SR}_2^2) -
\rho_{12}^2\hat{SR}_1\hat{SR}_2 -
(\gamma_{3,1}\hat{SR}_1 - \gamma_{3,2}\hat{SR}_2)\cdot 2\rho_{12}\right]$$

**Citation:** Jobson & Korkie (1981), "Performance Hypothesis Testing
with the Sharpe and Treynor Measures," *J. Finance*, 36(4); Memmel
(2003), "Performance Hypothesis Testing with the Sharpe Ratio,"
*Finance Letters*, 1(1).

### 9.4 White's Reality Check / SPA Test

White's RC: $\bar{V} = \max_k \sqrt{T} \cdot \bar{f}_k$, bootstrap the
distribution of the max statistic under the null. SPA (Hansen 2005)
studentizes the statistic for improved power.

**Citation:** White (2000), "A Reality Check for Data Snooping,"
*Econometrica*, 68(5); Hansen (2005), "A Test for Superior Predictive
Ability," *JBES*, 23(4).

### 9.5 PBO (Combinatorial Purged CV)

Generate combinatorially-paired train/test splits with purge and embargo
periods. For each pair, rank strategies by in-sample SR. PBO is the
probability that the best in-sample strategy ranks below the median
out-of-sample.

**Citation:** Bailey & López de Prado (2014), "Pseudo-Mathematics and
Financial Charlatanism," *Notices of the AMS*, 61(5); de Prado (2018,
Ch. 11–13).

### 9.6 Marginal Contribution to Portfolio Risk

$$\text{MCR}_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

where $\Sigma$ is the covariance matrix of strategy returns and $w$ the
weight vector.

**Citation:** Litterman (1996); Qian (2005), "Risk Parity and
Diversification," *J. of Investing*, 14(3).

### 9.7 Component VaR

$$\text{CVaR}_i = w_i \cdot \frac{\partial \text{VaR}}{\partial w_i}
= w_i \cdot (-\beta_i \cdot \text{VaR}_p)$$

**Assumption:** The formula above holds for parametric (linear) VaR
decomposition. When VaR is computed via the historical method (the
default per conventions table), the linear decomposition via beta is an
approximation. The implementation computes component VaR via marginal
revaluation: remove position $i$ and recompute VaR, then take the
difference.

**Citation:** Jorion (2006, *Value at Risk*, 3rd ed., Ch. 7);
Litterman (1996).

---

## 10. `report`

Separate install extra (`stratstat[report]`). Depends on `core` only —
never the reverse. No formulas; specification of visualization outputs.

| # | Output | Description |
|---|--------|-------------|
| 1 | Single-strategy tear sheet | Equity curve, drawdown chart with recovery shading, monthly heatmap, stats table |
| 2 | Multi-strategy comparison dashboard | Overlay equity curves, rolling-metric charts, correlation heatmap, ranking table |
| 3 | Rolling-metric charts | Any registered metric via `rolling()`, rendered as time series |
| 4 | Drawdown chart | Underwater equity curve with recovery-period shading |
| 5 | Cumulative return chart | Cumulative return index, optionally with benchmark overlay |
| 6 | Benchmark comparison overlay | Strategy vs. benchmark cumulative return and active return |
| 7 | Trade markers (MFE/MAE) | Optional intra-trade MFE/MAE overlay on the equity curve |
| 8 | Monthly heatmap matrix | Years × months return grid with color scale |
| 9 | Export formats | Interactive HTML, static PNG/SVG, Markdown table, LaTeX table, JSON serialization of `MetricSet` |

---

## References

Acerbi, C. & Tasche, D. (2002). "On the Coherence of Expected Shortfall." *JBF*, 26(7).
Adelman, M. A. (1969). "Comment on the 'H' Concentration Measure." *Rev. Econ. Stat.*, 51(1).
Ang, A. (2014). *Asset Management: A Systematic Approach to Factor Investing*. Oxford.
Ang, A. & Bekaert, G. (2004). "How Regimes Affect Asset Allocation." *FAJ*, 60(2).
Artzner, P. et al. (1999). "Coherent Measures of Risk." *Mathematical Finance*, 9(3).
Asness, C., Frazzini, A. & Pedersen, L. H. (2014). "Low-Risk Investing Without Industry Bets." *FAJ*, 70(4).
Bacon, C. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd ed. Wiley.
Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *J. Risk*, 15(2).
— (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias." *JPM*, 40(5).
— (2014). "Pseudo-Mathematics and Financial Charlatanism." *Notices of the AMS*, 61(5).
Basel Committee on Banking Supervision (1996). *Amendment to the Capital Accord to Incorporate Market Risks*.
Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton.
Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd ed. Duxbury.
CFA Institute. *CFA Program Curriculum* (current edition, Quantitative Methods, Vol. 1).
Choueifaty, Y. & Coignard, Y. (2008). "Toward Maximum Diversification." *JPM*, 35(1).
Connor, G., Goldberg, L. R. & Korajczyk, R. A. (2010). *Portfolio Risk Analysis*. Princeton.
Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton.
Damodaran, A. (2012). *Investment Valuation*, 3rd ed. Wiley.
de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
Efron, B. & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall.
Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.
Everitt, B. S. & Skrondal, A. (2010). *The Cambridge Dictionary of Statistics*, 4th ed. Cambridge.
Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
— (1930). "The Moments of the Distribution for Normal Samples." *Proc. London Math. Soc.*, s2-30(1).
Goodwin, T. H. (1998). "The Information Ratio." *FAJ*, 54(4).
Greene, W. H. (2018). *Econometric Analysis*, 8th ed. Pearson.
Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *JBES*, 23(4).
Hill, B. M. (1975). "A Simple General Approach to Inference About the Tail." *Ann. Stat.*, 3(5).
Hirschman, A. O. (1964). "The Paternity of an Index." *AER*, 54(5).
Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter and Quantile Estimation for the GPD." *Technometrics*, 29(3).
Hyndman, R. J. & Fan, Y. (1996). "Sample Quantiles in Statistical Packages." *The American Statistician*, 50(4).
Jarque, C. M. & Bera, A. K. (1987). "A Test for Normality." *Int. Stat. Rev.*, 55(2).
Jensen, M. C. (1968). "The Performance of Mutual Funds 1945–1964." *J. Finance*, 23(2).
Jobson, J. D. & Korkie, B. M. (1981). "Performance Hypothesis Testing with Sharpe and Treynor." *J. Finance*, 36(4).
Johnson, R. A. & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis*, 6th ed. Pearson.
Jorion, P. (2006). *Value at Risk*, 3rd ed. McGraw-Hill.
Kaplan, P. D. & Knowles, J. A. (2004). "Kappa: A Generalized Downside Risk-Adjusted Performance Measure." *JPM*.
Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *JPM*, 6(3).
Kelly, J. L. (1956). "A New Interpretation of Information Rate." *BSTJ*, 35(4).
Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *Ann. Stat.*, 17(3).
Ledoit, O. & Wolf, M. (2008). "Robust Performance Hypothesis Testing with the Sharpe Ratio." *J. Empirical Finance*, 15(5).
Litterman, R. (1996). "Hot Spots and Hedges." *JPM*, Special Issue.
Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *FAJ*, 58(4).
Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio." *Finance Letters*, 1(1).
Morningstar (2020). *Morningstar Performance Reporting Methodology*.
Pearson, K. (1895). "Note on Regression and Inheritance in the Case of Two Parents." *Proc. Royal Society*, 58.
— (1896). "Mathematical Contributions to the Theory of Evolution. III." *Phil. Trans. Royal Society A*, 187.
Perold, A. F. (1988). "The Implementation Shortfall: Paper versus Reality." *JPM*, 14(3).
Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Ann. Stat.*, 3(1).
Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." *Econometric Reviews*, 23(1).
Pospisil, J. & Vecer, J. (2011). "Maximum Drawdown of a Brownian Motion." *J. Applied Probability*, 48(3).
Qian, E. (2005). "Risk Parity and Diversification." *J. of Investing*, 14(3).
Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *J. Risk*, 2(3).
Roll, R. (1992). "A Mean/Variance Analysis of Tracking Error." *JPM*, 18(4).
Schwager, J. D. (1995). *Schwager on Futures: Technical Analysis*. Wiley.
Sharpe, W. F. (1964). "Capital Asset Prices." *J. Finance*, 19(3).
— (1966). "Mutual Fund Performance." *J. Business*, 39(1).
— (1994). "The Sharpe Ratio." *JPM*, 21(1).
Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *J. Investing*, 3(3).
Sortino, F. A. & van der Meer, R. (1991). "Downside Risk." *JPM*, 17(4).
Sweeney, R. J. (1988). "The Maximum Favorable Excursion Methodology." *JPM*.
Tharp, V. K. (1998). *Trade Your Way to Financial Freedom*. McGraw-Hill.
Thorp, E. O. (1997). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." *Handbook of Asset and Liability Management*.
Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *HBR*, 43(1).
Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
van Hemert, O. et al. (2020). *Tactical Asset Allocation*. (Drawdown analysis, Ch. 5.)
Vince, R. (1990). *Portfolio Management Formulas*. Wiley.
White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5).
Young, T. W. (1991). "Calmar Ratio: A Smoother Tool." *Futures*, 20(10).
Zangari, P. (1996). "A VaR Methodology for Portfolios That Include Options." *RiskMetrics Monitor*, Q1.
Zivot, E. & Wang, J. (2006). *Modeling Financial Time Series with S-PLUS*, 2nd ed. Springer.
