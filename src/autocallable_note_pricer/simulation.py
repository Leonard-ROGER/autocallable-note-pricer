import numpy as np

def simulate_st_paths(S0, r, d, T, sigma, obs_time, n_paths, seed):
    
    """
    Simulate the underlying level at each observation date using Black-Scholes dynamics,
    with antithetic variates.

    The parameter d is a continuous yield deducted from the underlying, so the
    risk-neutral drift is r - d. It covers two cases:
        - price index (e.g. Euro Stoxx 50): d is the fix dividend yield, as dividends
          are paid out and leave the index
        - decrement index (e.g. iStoxx 50 EW Decrement 5%): d is the decrement, as
          dividends are reinvested in the index and a fixed percentage is deducted
          each year instead

    Parameters
    ----------
    S0 : float
        Spot level of the underlying at t=0
    r : float
        Risk-free interest rate (continuous compounding)
    d : float
        Dividend yield (price index) or decrement (decrement index), per year
    sigma : float
        Volatility of the underlying
    T : float
        Maturity of the product in years
    dt : float
        Time between two observation dates in years (1 = annual, 0.25 = quarterly)
    n_paths : int
        Number of Monte Carlo paths, must be even (antithetic variates)
    seed : int
        Seed of the random generator, so that two calls with the same seed
        use the same random draws (needed for the coupon solver and the Greeks)

    Returns
    -------
    np.ndarray of shape (n_paths, int(round(T / dt)))
        Simulated levels of the underlying
        Each row is one path, each column is one observation date
    """
    
    if n_paths % 2 != 0: 
        raise ValueError("n_paths must be even (antithetic variates)")
    
    rng =np.random.default_rng(seed)
    
    nb_obs = int(round(T/obs_time))
    
    epsilon = rng.standard_normal((n_paths//2,nb_obs))
    epsilon = np.concatenate((epsilon, -epsilon), axis=0)
    
    log_returns = (r - d - 0.5 * (sigma ** 2)) * obs_time + sigma * np.sqrt(obs_time) * epsilon
    
    St = S0 * np.exp(np.cumsum(log_returns, axis=1))    
    
    return St

if __name__ == "__main__":
    S = simulate_st_paths(100, 0.03, 0.05, 10, 0.20, 1, 100000, 42)
    print(S.shape)
    print(S[:, -1].mean())