"""test_pricer checks for the autocallable note pricer.

Each test prices the base case (10y annual Phoenix on a 5% decrement index)
with 20,000 paths and changes one or two parameters to compare the result
with a known value or an expected direction.

Run with ``uv run pytest``.
"""

import math

from autocallable_note_pricer.autocall import price_autocallable_note

BASE = {
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
    "coupon_rate": 0.08,
    "memory": True,
    "nominal": 100,
    "n_paths": 20_000,
    "seed": 42,
}


def price_base(modifications: dict):
    """
    Price the base-case autocallable note with some parameters changed.

    Steps:
        1. Check that every key of modifications is a base-case parameter
        2. Copy the base-case parameters (BASE), replacing those listed in
           modifications, so that BASE is never modified
        3. Call price_autocallable_note with the resulting parameters

    Base case: 10-year Phoenix note on a 5% decrement index, annual observations,
    autocall barrier 100%, coupon barrier 70% with memory, PDI 60% (strike 100%),
    coupon 8% per year, S0 = S_ref = 100, sigma = 20%, r = 3%, no credit spread,
    20,000 paths, seed 42.

    Parameters
    ----------
    modifications : dict
        Parameters to override, keyed by argument name of
        price_autocallable_note (e.g. {"sigma": 0.30}).
        Parameters not listed keep their base-case value.
        Use {} to price the base case itself.

    Returns
    -------
    tuple of 8 elements
        Output of price_autocallable_note, in the same order:
        price, standard_error, pv_coupons, pv_nominal, pv_pdi_loss,
        exit_probabilities, capital_loss_probability, expected_life

    Raises
    ------
    KeyError
        If a key of modifications is not a parameter of BASE
    """
    for key in modifications:
        if key not in BASE:
            raise KeyError("Unknown parameter")

    new_base = {}
    for key in BASE:
        if key in modifications:
            new_base[key] = modifications[key]
        else:
            new_base[key] = BASE[key]
    return price_autocallable_note(**new_base)


def test_price_when_autocall_is_certain_at_year_one():
    """
    Price of the note when the autocall at year 1 is certain.

    With autocall and coupon barriers at 0% of S_ref, the note is called at
    year 1 on every path and pays nominal + one coupon. The coupon barrier is
    also lowered so that the coupon is paid on every path.
    All paths give the same cash flow, so there is no Monte Carlo noise:
        price = (nominal + coupon_rate * obs_time * nominal) * exp(-r * obs_time)
              = 108 * exp(-0.03) ≈ 104.81

    Checks:
        - price equal to the theoretical value
        - exit probability at year 1 equal to 1
    """

    (price, _, _, _, _, _, _, expected_life) = price_base(
        {
            "autocall_barrier": 0,
            "coupon_barrier": 0,
        }
    )

    expected_price = (
        BASE["nominal"] + BASE["coupon_rate"] * BASE["nominal"] * BASE["obs_time"]
    ) * math.exp(-BASE["r"] * BASE["obs_time"])

    assert abs(price - expected_price) <= 0.001
    assert abs(expected_life - 1) <= 0.001


def test_price_when_note_is_a_zero_coupon():
    """
    Price of the note when it behaves as a zero-coupon bond.

    With no coupon, an autocall barrier never reached (math.inf) and a PDI
    barrier never breached (0, the performance is always strictly positive),
    the note pays the nominal at maturity on every path and nothing else.
    All paths give the same cash flow, so there is no Monte Carlo noise:
        price = nominal * exp(-r * T)
              = 100 * exp(-0.3) ≈ 74.08

    Checks:
        - price equal to the theoretical value
        - expected life equal to T
        - capital loss probability equal to 0
    """

    (price, _, _, _, _, _, capital_loss_probability, expected_life) = price_base(
        {"autocall_barrier": math.inf, "coupon_rate": 0, "pdi_barrier": 0}
    )

    assert abs(expected_life - BASE["T"]) <= 0.001

    expected_price = BASE["nominal"] * math.exp(-BASE["r"] * BASE["T"])
    assert abs(price - expected_price) <= 0.001

    assert capital_loss_probability <= 0.001
