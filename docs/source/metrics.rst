Metric Inventory
================

StratStat ships **176 registered metrics** across 8 categories. Every metric
returns :class:`~stratstat.results.MetricResult` with a ``.meta["ref"]`` field
citing the formula's source. The full formulas live in the
:doc:`formula-reference` page.

.. list-table::
   :header-rows: 1

   * - Category
     - Tier (``requires``)
     - Count
     - Examples
   * - Descriptive
     - returns
     - 27
     - CAGR, volatility, skewness, kurtosis, Hurst exponent, percentiles
   * - Risk
     - returns
     - 24
     - VaR, CVaR, max drawdown, drawdown duration, ulcer index, tail ratio, EVT
   * - Risk adjusted
     - returns
     - 23
     - Sharpe, Sortino, Calmar, Omega, Kappa 3, Martin, gain to pain
   * - Inference
     - returns
     - 14
     - PSR, DSR, Lo's SE, Sharpe CI, Monte Carlo, block bootstrap CI
   * - Exposure
     - exposure
     - 24
     - gross and net exposure, HHI, turnover, leverage, beta, position coverage
   * - Trades
     - trades
     - 37
     - win rate, profit factor, expectancy, streaks, MFE and MAE, holding periods
   * - Benchmark
     - benchmark
     - 20
     - tracking error, information ratio, alpha, beta, information coefficient, capture
   * - Compare
     - compare
     - 7
     - Sharpe difference test, marginal risk contribution, diversification ratio, PBO

Descriptive (27)
----------------

``cagr``, ``annualized_volatility``, ``cumulative_return``,
``arithmetic_mean_return``, ``geometric_mean_return``, ``skewness``,
``excess_kurtosis``, ``best_period``, ``worst_period``,
``positive_period_ratio``, ``negative_period_ratio``, ``autocorrelation``,
``variance``, ``return_range``, ``percentiles``, ``coefficient_of_variation``,
``outlier_iqr``, ``stability``, ``hurst_exponent``, ``fractal_dimension``,
``consecutive_wins_losses``, ``exposure_time``, ``avg_up_period``,
``avg_down_period``, ``period_profit_factor``, ``period_payoff_ratio``,
``period_kelly_criterion``

Risk (24)
---------

``max_drawdown``, ``longest_drawdown_duration``, ``time_to_recovery``,
``average_drawdown``, ``average_drawdown_duration``, ``ulcer_index``,
``downside_deviation``, ``upside_deviation``, ``downside_semivariance``,
``var``, ``modified_var``, ``cvar``, ``tail_ratio``, ``common_sense_ratio``,
``hill_tail_index``, ``gpd_tail_fit``, ``risk_of_ruin``,
``drawdown_volatility``, ``drawdown_periods_count``, ``current_drawdown``,
``current_drawdown_duration``, ``drawdown_total_duration``, ``pain_index``,
``prospect_ratio``

Risk adjusted (23)
------------------

``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``, ``omega_ratio``,
``sterling_ratio``, ``burke_ratio``, ``kappa_3``, ``martin_ratio``,
``gain_to_pain_ratio``, ``pain_ratio``, ``recovery_factor``, ``k_ratio``,
``serenity_ratio``, ``upi``, ``modified_sharpe_ratio``,
``upside_potential_ratio``, ``risk_return_ratio``, ``roys_safety_first``,
``autocorr_penalty``, ``smart_sharpe``, ``smart_sortino``,
``adjusted_sortino_ratio``, ``rar``

Inference (14)
--------------

``jarque_bera``, ``psr``, ``dsr``, ``lo_sharpe_se``, ``sharpe_ci_analytic``,
``sharpe_ci_bootstrap``, ``min_track_record_length``, ``block_bootstrap_ci``,
``bias_ratio``, ``skewness_adjusted_sharpe``, ``probabilistic_sortino_ratio``,
``probabilistic_adjusted_sortino_ratio``, ``monte_carlo_distribution``,
``monte_carlo_probabilities``

Benchmark (20)
--------------

``alpha``, ``beta``, ``r_squared``, ``tracking_error``, ``information_ratio``,
``up_capture``, ``down_capture``, ``up_down_capture``, ``correlation``,
``active_return``, ``batting_average``, ``treynor_ratio``, ``outperformance``,
``outperformance_ratio``, ``underperforming_periods``, ``max_outperformance``,
``max_underperformance``, ``benchmark_volatility``, ``information_coefficient``,
``directional_consistency``

Exposure (24)
-------------

``gross_exposure``, ``net_exposure``, ``leverage``, ``long_exposure_pct``,
``short_exposure_pct``, ``long_book_return``, ``short_book_return``,
``long_beta``, ``short_beta``, ``position_concentration``,
``effective_n_positions``, ``turnover``, ``avg_holding_weight``,
``position_coverage``, ``long_position_coverage``, ``short_position_coverage``,
``exposure_volatility``, ``net_exposure_volatility``, ``exposure_cv``,
``exposure_utilization``, ``exposure_directional_bias``,
``exposure_percentiles``, ``period_counts``, ``active_share``

Trades (37)
-----------

``total_trades``, ``win_rate``, ``win_rate_long``, ``win_rate_short``,
``avg_win``, ``avg_loss``, ``win_loss_ratio``, ``profit_factor``,
``expectancy``, ``avg_holding_period``, ``holding_period_distribution``,
``max_consecutive_wins``, ``max_consecutive_losses``, ``pnl_distribution``,
``implementation_shortfall``, ``best_trade``, ``worst_trade``,
``avg_winning_duration``, ``avg_losing_duration``, ``payoff_ratio``,
``cpc_ratio``, ``sqn``, ``trade_duration_std``, ``trade_return_std``,
``geometric_mean_return_per_trade``, ``outlier_win_ratio``,
``outlier_loss_ratio``, ``mfe``, ``mae``, ``kelly_criterion``,
``long_short_trade_count``, ``long_short_trade_pct``,
``long_short_winning_losing``, ``long_short_avg_duration``,
``long_short_total_pnl``, ``long_short_avg_pnl``, ``long_short_best_worst``

Compare (7)
-----------

``correlation_matrix``, ``diversification_ratio``, ``sharpe_difference_test``,
``whites_reality_check``, ``pbo``, ``marginal_contribution_to_risk``,
``component_var``

Discovering metrics at runtime
------------------------------

.. code-block:: python

   import stratstat as ss

   for m in ss.list_metrics():
       print(m["name"], m["category"], m["requires"])

   # Filter by category
   for m in ss.list_metrics(category="risk"):
       print(m["name"])

For the full API reference with signatures and docstrings, see the :doc:`api`
page.
