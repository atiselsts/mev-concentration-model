This model describes how builder/solver concentration in block production changes as a function of block time.
It's inspired by this discussion on X: https://x.com/RyanSAdams/status/1991896909955604816

Model assumptions:

- A builder may have two forms of timing advantage:
  1. Absolute advantage: a fixed latency improvement due to physical proximity
     (e.g., being 300 km closer to the sequencer yields ~1 ms advantage).
  2. Relative advantage: faster block construction or processing
     (e.g., a more optimized algorithm provides a consistent ~10% speedup).

- The block production mechanism is FCFS (first-come, first-served).

- All timing processes are stochastic. Both forms of advantage include variance;
  neither is treated as a deterministic offset.

- We do not model how MEV revenue is split between builders and searchers.
  Only the distribution of block wins across builders matters.

- The MEV side of the model is intentionally configured to show a worst-case
  concentration scenario for Ethereum L1:
  * 90% of total MEV revenue is assumed to come from non-atomic arbitrage.
  * Non-atomic arbitrage revenue per block is assumed to scale as t * sqrt(t)
    (i.e., proportional to t^1.5).

Comments on the last assumption:
- The sqrt(t) dependence comes from treating non-atomic arbitrage as the time
  value of a "free option." Empirical work shows that real non-atomic AMM
  arbitrage revenue often scales more weakly—particularly at short block
  intervals.
- Not all chains today, and likely not future Ethereum designs, will have MEV
  dominated by non-atomic arbitrage. Improved AMM mechanisms reduce this effect.

Applicability:
- Because this model uses FCFS ordering, its results do not directly apply to
  systems with priority fees, tips, Timeboost, or similar mechanisms.
  In such systems, a latency advantage changes a bidder's ability to calibrate
  their bid, and a bidder wins only when doing so is economically optimal.
  This reduces revenue relative to FCFS but may increase profit.

Normalization:
- All MEV values are shown on a relative scale in which total MEV revenue per
  second is equal to 1.0 when the block time is exactly 1 second.
