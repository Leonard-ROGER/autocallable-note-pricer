import numpy as np
from autocallable_note_pricer.simulation import simulate_st_paths

def autocallable_note_payoff(S, S_ref, obs_time, autocall_barrier, coupon_barrier, pdi_barrier, pdi_strike, coupon_rate, memory, nominal):
    
    """
    Compute the cash flows of an autocallable note (Phoenix or Athena) on each
    simulated path, at each observation date, without discounting.

    At each observation date, for a note still alive:
        1. Coupon: paid if perf >= coupon_barrier. With memory, missed coupons
           are paid back at the same time.
        2. Autocall (all dates except the last one): if perf >= autocall_barrier,
           the nominal is repaid and the note ends.
        3. Maturity (last date): the nominal is repaid; if perf < pdi_barrier, the
           investor bears the loss of a put with strike pdi_strike.

    Product types:
        - Phoenix: coupon_barrier < autocall_barrier, with or without memory
        - Athena: coupon_barrier = autocall_barrier and memory = True

    Parameters
    ----------
    S : np.ndarray of shape (n_paths, n_obs)
        Simulated levels of the underlying at each observation date
    S_ref : float
        Initial reference level, used to compute performances and barriers
    obs_time : float
        Time between two observation dates in years (1 = annual, 0.25 = quarterly)
    autocall_barrier : float
        Autocall barrier as a percentage of S_ref (e.g. 1.00)
    coupon_barrier : float
        Coupon barrier as a percentage of S_ref (e.g. 0.70)
    pdi_barrier : float
        Capital protection barrier at maturity as a percentage of S_ref (e.g. 0.60)
    pdi_strike : float
        Strike of the down-and-in put as a percentage of S_ref (usually 1.00)
    coupon_rate : float
        Annual coupon rate (e.g. 0.08 for 8% per year)
    memory : bool
        If True, missed coupons are paid back when the coupon condition is met again
    nominal : float
        Nominal amount invested (e.g. 100)

    Returns
    -------
    coupons : np.ndarray of shape (n_paths, n_obs)
        Coupons paid at each date
    reimburse_nominal : np.ndarray of shape (n_paths, n_obs)
        Nominal repaid at the autocall date or at maturity
    pdi_loss : np.ndarray of shape (n_paths, n_obs)
        Capital loss at maturity (negative or zero)
    exit_indice : list of int, length n_paths
        Index of the date when the note ends (autocall or maturity)
    capital_loss : list of bool, length n_paths
        True if the investor loses capital at maturity

    The total cash flow on each path is coupons + reimburse_nominal + pdi_loss.
    """
    
    perf = S/S_ref
    
    coupon = coupon_rate * obs_time * nominal
    
    coupons = np.zeros(perf.shape)
    reimburse_nominal = np.zeros(perf.shape)
    pdi_loss = np.zeros(perf.shape)
    
    autocallable_note_life = [True for i in range(len(perf))]
    lost_coupons = [0 for _ in range(len(perf))]
    exit_indice = [len(perf[0]) - 1 for i in range(len(perf))]
    capital_loss = [False for _ in range(len(perf))]
    
    for i in range(len(perf)):
        for j in range(len(perf[0])):
            if autocallable_note_life[i]:
                if perf[i][j] >= coupon_barrier:
                    coupons[i][j] = coupon * (lost_coupons[i] + 1)
                    lost_coupons[i] = 0
                
                else :
                    if memory:
                        lost_coupons[i] += 1
                
            
                if j == len(perf[0]) - 1:
                    reimburse_nominal[i][j] = nominal
                    if perf[i][j] < pdi_barrier:
                        pdi_loss[i][j] = -1 * (pdi_strike - perf[i][j]) * nominal
                        capital_loss[i] = True
                else :
                    if perf[i][j] >= autocall_barrier:
                        reimburse_nominal[i][j] = nominal
                        exit_indice[i] = j
                        autocallable_note_life[i] = False

    return coupons, reimburse_nominal, pdi_loss, exit_indice, capital_loss

def price_autocallable_note(S0, S_ref, r, d, sigma, T, obs_time, credit_spread,
                            autocall_barrier, coupon_barrier, pdi_barrier, pdi_strike,
                            coupon_rate, memory, nominal, n_paths, seed):
    
    """
    Price an autocallable note (Phoenix or Athena) by Monte Carlo simulation.

    Steps:
        1. Simulate the underlying at each observation date (antithetic variates)
        2. Compute the cash flows of the note on each path (coupons, nominal, PDI loss)
        3. Discount each cash flow at its date with the risk-free rate plus the
           issuer credit spread: DF(t) = exp(-(r + credit_spread) * t)
        4. Average the present values over all paths

    The price is decomposed into three building blocks:
        price = PV(coupons) + PV(nominal) + PV(PDI loss)

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
    coupon_rate : float
        Annual coupon rate (e.g. 0.08 for 8% per year)
    memory : bool
        If True, missed coupons are paid back when the coupon condition is met again
    nominal : float
        Nominal amount invested (e.g. 100)
    n_paths : int
        Number of Monte Carlo paths, must be even (antithetic variates)
    seed : int
        Seed of the random generator (same seed = same paths)

    Returns
    -------
    tuple of 8 elements, in this order:
        price : float
            Monte Carlo price of the note
        standard_error : float
            Standard error of the price, computed on antithetic pair averages
        pv_coupons : float
            Present value of the coupons
        pv_nominal : float
            Present value of the nominal repaid (autocall or maturity)
        pv_pdi_loss : float
            Present value of the capital loss at maturity (negative or zero)
        exit_probabilities : np.ndarray of shape (n_obs,)
            Probability that the note ends at each observation date
            (the last element is the probability of reaching maturity)
        capital_loss_probability : float
            Probability that the investor loses capital at maturity
        expected_life : float
            Expected life of the note in years
    """
    
    S = simulate_st_paths(S0, r, d, T, sigma, obs_time, n_paths, seed)
    
    coupons, reimburse_nominal, pdi_loss, exit_indice, capital_loss = autocallable_note_payoff(S, S_ref, obs_time, autocall_barrier, 
                                                                                               coupon_barrier, pdi_barrier, pdi_strike, 
                                                                                               coupon_rate, memory, nominal)

    discount_factors = np.zeros(coupons.shape)
    for i in range(len(discount_factors)):
        for j in range(len(discount_factors[0])):
            discount_factors[i][j] = np.exp(-1 * (r + credit_spread) * (j+1) * obs_time)
    
    present_value_coupons = coupons * discount_factors
    present_value_reimburse_nominal = reimburse_nominal * discount_factors
    present_value_pdi_loss = pdi_loss * discount_factors
    
    present_value = [0 for _ in range(len(coupons))]
    
    for i in range(len(coupons)):
        for j in range(len(coupons[i])):
            present_value[i] += present_value_coupons[i][j] + present_value_reimburse_nominal[i][j] + present_value_pdi_loss[i][j]
    present_value = np.array(present_value)
    
    price = np.sum(present_value) / len(present_value)
    
    present_value_first_half, present_value_antithetic = np.split(present_value,2)
    
    mean_present_value = (present_value_first_half + present_value_antithetic)/2
    
    standard_error = np.std(mean_present_value) / np.sqrt(n_paths//2)
    
    
    exit_probabilities = np.zeros(len(coupons[0]))
    for i in range(n_paths):
        exit_probabilities[exit_indice[i]] += 1 / n_paths
    
    expected_life = 0
    for i in range(len(exit_probabilities)):
        expected_life += (i+1)*exit_probabilities[i]*obs_time
    
    capital_loss_probabilities = 0
    for i in range(len(capital_loss)):
        if capital_loss[i]:
            capital_loss_probabilities += 1/n_paths
    
    present_value_coupons_mean = present_value_coupons.sum() / n_paths
    present_value_reimburse_nominal_mean = present_value_reimburse_nominal.sum() / n_paths
    present_value_pdi_loss_mean = present_value_pdi_loss.sum() / n_paths 
    
    return (float(price), float(standard_error), present_value_coupons_mean, present_value_reimburse_nominal_mean, 
            present_value_pdi_loss_mean, exit_probabilities, capital_loss_probabilities, expected_life)