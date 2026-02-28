from src.utils import calculate_currents, calculate_voltages

def synchronous_machine_equations_SM4(t, x, omega_B, H, P_m, X_q_dash, X_d_dash, D, T_d_dash, X_d, E_fd, T_q_dash, X_q, Vs, theta_vs):
    theta, omega, E_d_dash, E_q_dash = x
    I_d, I_q = calculate_currents(theta, E_d_dash, E_q_dash, X_q_dash, X_d_dash, Vs, theta_vs)
    dtheta_dt = omega 
    domega_dt = ( omega_B / (2 * H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (X_q_dash - X_d_dash) * I_q * I_d - D * omega )
    dE_q_dash_dt = (1 / T_d_dash) * (- E_q_dash - I_d * (X_d - X_d_dash) + E_fd)
    dE_d_dash_dt = (1 / T_q_dash) * (- E_d_dash + I_q * (X_q - X_q_dash))
    return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]

def synchronous_machine_equations_SM6(t, x, omega_B, H, P_m, X_q_dash, X_d_dash, D, T_d_dash, X_d, E_fd, T_q_dash, X_q, T_d_dash_dash, T_q_dash_dash, X_d_dash_dash, X_q_dash_dash, Vs, theta_vs):
    theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash = x
    I_d, I_q = calculate_currents(theta, E_d_dash, E_q_dash, X_q_dash, X_d_dash, Vs, theta_vs)
    dtheta_dt = omega 
    domega_dt = ( omega_B / (2 * H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (X_q_dash - X_d_dash) * I_q * I_d - D * omega )
    dE_q_dash_dt = (1 / T_d_dash) * (- E_q_dash - I_d * (X_d - X_d_dash) + E_fd)
    dE_d_dash_dt = (1 / T_q_dash) * (- E_d_dash + I_q * (X_q - X_q_dash))
    dE_q_dash_dash_dt = (1 / T_d_dash_dash) * (E_q_dash - E_q_dash_dash + I_d * (X_d_dash - X_d_dash_dash))
    dE_d_dash_dash_dt = (1 / T_q_dash_dash) * (E_d_dash - E_d_dash_dash - I_q * (X_q_dash - X_q_dash_dash))
    return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt]


def synchronous_machine_equations_SM_AVR_GOV(t, x, omega_B, H, X_q_dash, X_d_dash, D, T_d_dash, X_d, T_q_dash, X_q, T_F, K_F, T_A, K_A, V_ref, T_E, K_E, T_ch, T_sv, P_c, R_d, Vs, theta_vs):
    theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x
    I_d, I_q = calculate_currents(theta, E_d_dash, E_q_dash, X_q_dash, X_d_dash,Vs, theta_vs)
    V_t = calculate_voltages(theta, I_d, I_q, Vs , theta_vs)
    dtheta_dt = omega
    domega_dt = (omega_B / (2 * H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (X_q_dash - X_d_dash) * I_q * I_d - D * omega )
    dE_q_dash_dt = (1 / T_d_dash) * (- E_q_dash - I_d * (X_d - X_d_dash) + E_fd)
    dE_d_dash_dt = (1 / T_q_dash) * (- E_d_dash + I_q * (X_q - X_q_dash))
    dR_F_dt = (1 / T_F) * (-R_F + (K_F / T_F) * E_fd)
    V_r_min = 0.8
    V_r_max = 1.2
    dV_r_raw = (1 / T_A) * (-V_r + (K_A * R_F) - (K_A * K_F / T_F) * E_fd + K_A * (V_ref - V_t))
    
    if V_r <= V_r_min and dV_r_raw < 0:
        dV_r_dt = 0  # Stop decreasing if at the lower bound
        V_r = V_r_min
    elif V_r >= V_r_max and dV_r_raw > 0:
        dV_r_dt = 0  # Stop increasing if at the upper bound
        V_r = V_r_max
    else:
        dV_r_dt = dV_r_raw  # Use the raw derivative otherwise
    dE_fd_dt = (1 / T_E) * (-(K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
    dP_m_dt = (1 / T_ch) * (-P_m + P_sv)
    dP_sv_dt = (1 / T_sv) * (-P_sv + P_c - (1/R_d) * (omega/omega_B))
    return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]
