import numpy as np
import matplotlib.pyplot as plt

from autocallable_note_pricer.autocall import price_autocallable_note
from autocallable_note_pricer.solver import solve_coupon


def plot_coupon_vs_vol(
    S0,
    S_ref,
    r,
    d,
    T,
    obs_time,
    credit_spread,
    autocall_barrier,
    coupon_barrier,
    pdi_barrier,
    pdi_strike,
    memory,
    nominal,
    n_paths,
    seed,
    margin,
    n_point,
    vol_min,
    vol_max,
):
    """
    Plot the solved coupon and the probability of capital loss as a function of the volatility

    For each volatility, the coupon is solved with the same margin and the same seed,
    so the curves only move because of the volatility (no Monte Carlo noise between points).
    A higher volatility gives a higher coupon (the client sells volatility through the PDI)
    but also a higher probability of capital loss at maturity.

    Parameters
    ----------
    S0 : float
        Spot level of the underlying at t=0
    S_ref : float
        Initial reference level, used to compute performances and barriers
    r : float
        Risk-free interest rate (continuous compounding)
    d : float
        Dividend yield (price index) or decrement (decrement index), per year
    T : float
        Maturity of the note in years
    obs_time : float
        Time between two observation dates in years (1 = annual, 0.25 = quarterly)
    credit_spread : float
        Issuer credit spread added to r for discounting (0 = no issuer risk)
    autocall_barrier : float
        Autocall barrier as a percentage of S_ref (e.g. 1.00)
    coupon_barrier : float
        Coupon barrier as a percentage of S_ref (e.g. 0.70)
    pdi_barrier : float
        Capital protection barrier at maturity as a percentage of S_ref (e.g. 0.60)
    pdi_strike : float
        Strike of the down-and-in put as a percentage of S_ref (usually 1.00)
    memory : bool
        If True, missed coupons are paid back when the coupon condition is met again
    nominal : float
        Nominal amount invested (e.g. 100)
    n_paths : int
        Number of Monte Carlo paths, must be even (antithetic variates)
    seed : int
        Seed of the random generator (same seed = same paths)
    margin : float
        Total margin of the bank as a fraction of the nominal (e.g. 0.05)
    n_point : int
        Number of volatility values to evaluate
    vol_min : float
        Lower bound of the volatility range (e.g. 0.15)
    vol_max : float
        Upper bound of the volatility range (e.g. 0.35)
    """

    vol_array = np.linspace(vol_min, vol_max, n_point)

    coupon_array = np.zeros(n_point)
    capital_loss_array = np.zeros(n_point)

    for i in range(n_point):
        coupon_array[i] = solve_coupon(
            S0,
            S_ref,
            r,
            d,
            vol_array[i],
            T,
            obs_time,
            credit_spread,
            autocall_barrier,
            coupon_barrier,
            pdi_barrier,
            pdi_strike,
            memory,
            nominal,
            n_paths,
            seed,
            margin,
        )

        capital_loss_array[i] = price_autocallable_note(
            S0,
            S_ref,
            r,
            d,
            vol_array[i],
            T,
            obs_time,
            credit_spread,
            autocall_barrier,
            coupon_barrier,
            pdi_barrier,
            pdi_strike,
            coupon_array[i],
            memory,
            nominal,
            n_paths,
            seed,
        )[6]

        print(
            "sigma = "
            + str(round(vol_array[i] * 100, 1))
            + "% -> coupon = "
            + str(round(coupon_array[i] * 100, 2))
            + "%, capital loss probability = "
            + str(round(capital_loss_array[i] * 100, 1))
            + "%"
        )

    plt.figure()
    plt.plot(
        vol_array * 100,
        coupon_array * 100,
        marker="o",
        label="Solved coupon",
        color="blue",
    )
    plt.legend()
    plt.xlabel("Volatility of the underlying (%)")
    plt.ylabel("Annual coupon (%)")
    plt.title(
        "Autocallable Note Coupon as a Function of Volatility (margin "
        + str(round(margin * 100))
        + "%)"
    )
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure()
    plt.plot(
        vol_array * 100,
        capital_loss_array * 100,
        marker="o",
        label="Probability of capital loss",
        color="red",
    )
    plt.legend()
    plt.xlabel("Volatility of the underlying (%)")
    plt.ylabel("Probability of capital loss at maturity (%)")
    plt.title("Probability of Capital Loss as a Function of Volatility")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    plot_coupon_vs_vol(
        S0=100,
        S_ref=100,
        r=0.03,
        d=0.05,
        T=10,
        obs_time=1,
        credit_spread=0.0,
        autocall_barrier=1.0,
        coupon_barrier=0.7,
        pdi_barrier=0.6,
        pdi_strike=1.0,
        memory=True,
        nominal=100,
        n_paths=20_000,
        seed=42,
        margin=0.05,
        n_point=5,
        vol_min=0.15,
        vol_max=0.35,
    )
