import numpy as np

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
    lost_coupons = [0 for i in range(len(perf))]
    exit_indice = [len(perf[0]) - 1 for i in range(len(perf))]
    capital_loss = [False for i in range(len(perf))]
    
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

if __name__ == "__main__":
    S = np.array([[105, 110, 120],    # rappel dès l'année 1
                  [ 80,  65, 101],    # coupon an 1, rien an 2, coupon x2 + nominal an 3
                  [ 90,  75,  50]],   # coupons ans 1-2, perte de 50 % à maturité
                 dtype=float)

    coupons, nominal, pdi_loss, exit_idx, capital_loss = autocallable_note_payoff(
        S, S_ref=100, obs_time=1, autocall_barrier=1.00, coupon_barrier=0.70,
        pdi_barrier=0.60, pdi_strike=1.00, coupon_rate=0.08, memory=True, nominal=100)

    print("coupons :\n", coupons)
    print("nominal :\n", nominal)
    print("pdi_loss :\n", pdi_loss)
    print("exit_idx :", exit_idx)
    print("capital_loss :", capital_loss)