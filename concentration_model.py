#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence
from ing_theme_matplotlib import mpl_style

# Block timings to test (in seconds)
BLOCK_INTERVALS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

# These constants describe the absolute and relative advantages of builders
ADVANTAGE_IN_MILLISECONDS = 30.0
ADVANTAGE_IN_PERCENT = 3.0

# Standard deviations for absolute and relative advantages (as relative noise)
ABS_ADV_SIGMA = 0.1  # 10% - needs to be calibrated with real data
REL_ADV_SIGMA = 0.1  # 10% - needs to be calibrated with real data

# Share of non-atomic arbitrage in total MEV revenue per second at t = 1
RELATIVE_SHARE_OF_NONATOMIC_ARB = 0.9  # in [0.0, 1.0]

# Number of simulated blocks per configuration
N_BLOCKS = 100_000


def get_relative_mev_revenue_of_block(t: float) -> float:
    """
    Compute relative MEV revenue of a single block given its time t (in seconds).

    Model:
    - Non-atomic arbitrage revenue per second scales as ~ t**0.5
      (time value of a "free option").
    - Other MEV types scale linearly with time per block => constant per second.
    - We mix them linearly by RELATIVE_SHARE_OF_NONATOMIC_ARB.
    - Then multiply by t to get per-block revenue.

    Normalization:
    - At t = 1, total MEV per second is 1.0 by construction.
    """
    if t <= 0:
        raise ValueError("Block time t must be positive.")

    alpha = RELATIVE_SHARE_OF_NONATOMIC_ARB
    rev_per_second = alpha * (t ** 0.5) + (1.0 - alpha)
    return rev_per_second * t


class Builder:
    def __init__(self, name: str, absolute_advantage: float, relative_advantage: float) -> None:
        """
        :param name: Identifier for the builder.
        :param absolute_advantage: Latency advantage in seconds (positive = faster).
        :param relative_advantage: Relative speed advantage (e.g. 0.03 for 3% faster).
        """
        self.name = name
        self.absolute_advantage = absolute_advantage
        self.relative_advantage = relative_advantage
        self.reset()

    def reset(self) -> None:
        self.blocks_won: int = 0
        self.mev_revenue: float = 0.0

    def on_block_won(self, t: float) -> None:
        self.blocks_won += 1
        self.mev_revenue += get_relative_mev_revenue_of_block(t)

    def get_relative_block_processing_delay(self, t: float) -> float:
        """
        Returns a relative "delay score" for this builder on a block of length t,
        where smaller is better (i.e., more negative = faster).

        This is not a physical time, but a linearized proxy:
        - absolute_advantage contributes a fixed shift with noise.
        - relative_advantage contributes a shift proportional to t with noise.
        """
        delay = 0.0

        if self.absolute_advantage != 0.0:
            # Absolute advantage modeled as: -A * Normal(1, ABS_ADV_SIGMA)
            delay -= self.absolute_advantage * np.random.normal(1.0, ABS_ADV_SIGMA)

        # Relative advantage modeled as: -t * Normal(mu, sigma), mu = relative_advantage
        delay -= t * np.random.normal(self.relative_advantage, REL_ADV_SIGMA)

        return delay

    def __str__(self) -> str:
        return f"{self.name}: blocks_won={self.blocks_won} rev={self.mev_revenue:.3f}"


def compete(builders: Sequence[Builder], t: float) -> None:
    """
    Simulate competition for one block of duration t among the given builders.
    The builder with the smallest "delay" wins the block.
    """
    times = [(i, b.get_relative_block_processing_delay(t)) for i, b in enumerate(builders)]
    times.sort(key=lambda x: x[1])
    winner_idx = times[0][0]
    builders[winner_idx].on_block_won(t)


def evaluate_competition(builders: Sequence[Builder], t: float) -> List[float]:
    """
    Run N_BLOCKS simulations at fixed block time t, then normalize MEV revenue
    to be per second, and print per-builder stats.

    :return: List of MEV revenues per second for each builder.
    """
    for _ in range(N_BLOCKS):
        compete(builders, t)

    simulation_time_seconds = N_BLOCKS * t

    # Normalize MEV revenue to per-second
    for b in builders:
        b.mev_revenue /= simulation_time_seconds

    total_mev_revenue = sum(b.mev_revenue for b in builders)

    for b in builders:
        print(b)
        share = 100.0 * b.mev_revenue / total_mev_revenue if total_mev_revenue > 0 else 0.0
        print(f"  share={share:.2f}%")

    return [b.mev_revenue for b in builders]


def compare_across_block_times(msg: str, builders: Sequence[Builder]) -> None:
    """
    Compare builder MEV shares across different block intervals defined in BLOCK_INTERVALS.
    """
    print(msg)
    results: List[List[float]] = [[] for _ in builders]

    for t in BLOCK_INTERVALS:
        for b in builders:
            b.reset()

        num_blocks_per_second = 1.0 / t
        total_mev_per_second = get_relative_mev_revenue_of_block(t) * num_blocks_per_second
        print(f"t={t:.1f} total MEV revenue per second: {total_mev_per_second:.2f}")

        t_results = evaluate_competition(builders, t)
        for i, r in enumerate(t_results):
            results[i].append(r)
        print("")

    # Plot results: MEV revenue per second vs block interval for each builder
    colors = ["#E75A5A", "#ffb437", "#37b6ff", "#4A90E2"]
    def color_for(b: Builder, i: int):
        if b.absolute_advantage > 0:
            return colors[i % 2]
        return colors[2 + i % 2]

    # Choose markers based on colocated vs non-colocated
    def marker_for(b: Builder):
        return "s" if b.absolute_advantage > 0 else "o"

    plt.figure(figsize=(8, 5))
    for i, b in enumerate(builders):
        plt.plot(BLOCK_INTERVALS, results[i],
                 marker=marker_for(b),
                 label=b.name,
                 color=color_for(b, i))

    plt.xlabel("Block interval (seconds)")
    plt.ylabel("MEV revenue per second (relative)")
    plt.title(msg)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    filename = msg.replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.close()


    # Convert results into shares (each column sums to 1.0)
    # results[i][j] = MEV/sec for builder i at BLOCK_INTERVALS[j]
    shares = []
    for j in range(len(BLOCK_INTERVALS)):
        column = [results[i][j] for i in range(len(builders))]
        total = sum(column)
        shares.append([v / total * 100.0 for v in column])

    # shares[j][i] = percentage share of builder i at block interval j
    # transpose to builder-major lists
    shares_by_builder = list(map(list, zip(*shares)))

    labels = [b.name for b in builders]

    # Plot relative shares (%)
    plt.figure(figsize=(8, 5))

    all_colors = [color_for(b, i) for i, b in enumerate(builders)]
    # Create stacked area plot
    plt.stackplot(
        BLOCK_INTERVALS,
        shares_by_builder,
        colors=all_colors[:len(builders)],
        alpha=0.85,
        labels=labels
    )

    plt.xlabel("Block interval (seconds)")
    plt.ylabel("MEV share (%)")
    plt.title(msg)
    plt.grid(False) #, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    filename = "relative_" + msg.replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.close()


def main():
    mpl_style(True)
    np.random.seed(42)

    advantage_seconds = ADVANTAGE_IN_MILLISECONDS / 1000.0
    advantage_fraction = ADVANTAGE_IN_PERCENT / 100.0

    # Various types of builders
    optimized_colocated = Builder("optimized colocated", advantage_seconds, advantage_fraction)
    colocated = Builder("colocated", advantage_seconds, 0.0)
    optimized = Builder("optimized", 0.0, advantage_fraction)
    baseline = Builder("baseline", 0.0, 0.0)
    all_builders = (optimized_colocated, colocated, optimized, baseline)

    compare_across_block_times(
        "Colocated builder vs non-colocated builder",
        (colocated, baseline),
    )

    compare_across_block_times(
        "Colocated builder vs optimized builder",
        (colocated, optimized),
    )

    compare_across_block_times("All builders", all_builders)


if __name__ == "__main__":
    main()
