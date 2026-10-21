from autocallable_note_pricer.autocall import price_autocallable_note


def solve_coupon(
    S0,
    S_ref,
    r,
    d,
    sigma,
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
    coupon_low=0.0,
    coupon_high=0.30,
    tolerance=1e-4,
    max_iter=100,
):
    """
    Find the annual coupon rate of the autocallable note by bisection.

    Steps:
        1. Compute the target price: nominal * (1 - margin)
           (the client pays the nominal, the bank keeps the margin)
        2. Check that the target price is between the prices at coupon_low
           and coupon_high (the price increases with the coupon)
        3. Bisection: price the note at the middle coupon and keep the half
           interval that contains the target price, until the interval is
           smaller than tolerance
        4. Return the middle of the final interval

    Every pricing uses the same seed, so the paths are identical and the price
    only changes because of the coupon (no Monte Carlo noise between iterations).

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
    sigma : float
        Volatility of the underlying
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
        Total margin of the bank as a fraction of the nominal
        (e.g. 0.05 for 5 points, target price 95 for a nominal of 100)
    coupon_low : float
        Lower bound of the search interval for the annual coupon (e.g. 0.0)
    coupon_high : float
        Upper bound of the search interval for the annual coupon (e.g. 0.30)
    tolerance : float
        Width of the final interval on the coupon (1e-4 = 0.01% per year)
    max_iter : int
        Maximum number of iterations (safety stop)

    Returns
    -------
    coupon : float
        Annual coupon rate that gives the target price (e.g. 0.0737)

    Raises
    ------
    ValueError
        If margin is not in [0, 1[, or if the target price is not reached
        for a coupon between coupon_low and coupon_high
    """
    if margin < 0 or margin >= 1:
        raise ValueError("Margin must be in [0, 1[")

    target_price = nominal * (1 - margin)

    # Prices at the bounds of the search interval
    (price_low, _, _, _, _, _, _, _) = price_autocallable_note(
        S0,
        S_ref,
        r,
        d,
        sigma,
        T,
        obs_time,
        credit_spread,
        autocall_barrier,
        coupon_barrier,
        pdi_barrier,
        pdi_strike,
        coupon_low,
        memory,
        nominal,
        n_paths,
        seed,
    )
    (price_high, _, _, _, _, _, _, _) = price_autocallable_note(
        S0,
        S_ref,
        r,
        d,
        sigma,
        T,
        obs_time,
        credit_spread,
        autocall_barrier,
        coupon_barrier,
        pdi_barrier,
        pdi_strike,
        coupon_high,
        memory,
        nominal,
        n_paths,
        seed,
    )

    # The price increases with the coupon: the target must be in between
    if target_price < price_low or target_price > price_high:
        raise ValueError("Target price not reached between coupon_low and coupon_high")

    low = coupon_low
    high = coupon_high
    nb_iter = 0

    while high - low >= tolerance and nb_iter < max_iter:
        mid = (low + high) / 2

        (price_mid, _, _, _, _, _, _, _) = price_autocallable_note(
            S0,
            S_ref,
            r,
            d,
            sigma,
            T,
            obs_time,
            credit_spread,
            autocall_barrier,
            coupon_barrier,
            pdi_barrier,
            pdi_strike,
            mid,
            memory,
            nominal,
            n_paths,
            seed,
        )

        if price_mid > target_price:
            high = mid  # coupon too generous: the note is worth too much
        else:
            low = mid  # coupon too low: the note is worth too little

        nb_iter += 1

    coupon = (low + high) / 2
    return coupon


if __name__ == "__main__":
    params = {
        "S0": 100,
        "S_ref": 100,
        "r": 0.03,
        "d": 0.05,
        "sigma": 0.20,
        "T": 10,
        "obs_time": 1,
        "credit_spread": 0.0,
        "autocall_barrier": 1.0,
        "coupon_barrier": 0.7,
        "pdi_barrier": 0.6,
        "pdi_strike": 1.0,
        "memory": True,
        "nominal": 100,
        "n_paths": 100_000,
        "seed": 42,
    }
    margin = 0.05

    coupon = solve_coupon(**params, margin=margin)
    print(f"Coupon solved: {coupon:.2%} per year")

    # Check: the price at the solved coupon must be close to the target
    price = price_autocallable_note(**params, coupon_rate=coupon)[0]
    print(f"Price at this coupon: {price:.4f} (target {100 * (1 - margin):.2f})")
