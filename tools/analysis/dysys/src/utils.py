import numpy as np

def calculate_currents( theta, E_d_dash, E_q_dash, X_q_dash, X_d_dash, Vs, theta_vs):
    
    """
    Calculates the currents I_d and I_q based on the given parameters.

    Parameters:
    theta (rad): The angle .
    E_d_dash (pu): The value of E_d_dash.
    E_q_dash (pu): The value of E_q_dash.

    Returns:
    tuple: A tuple containing the calculated values of I_d and I_q.
    """

    Rs=0.0
    Re=0.0
    Xep=0.1
    
    alpha = [[(Rs+Re), -(X_q_dash+Xep)], [(X_d_dash+Xep), (Rs+Re)]]
    beta = [[E_d_dash - Vs*np.sin(theta-theta_vs)], [E_q_dash - Vs*np.cos(theta-theta_vs)]]
    inv_alpha = np.linalg.inv(alpha)
    # Calculate I_d and I_q
    I_d= inv_alpha[0][0]*beta[0][0] + inv_alpha[0][1]*beta[1][0]
    I_q= inv_alpha[1][0]*beta[0][0] + inv_alpha[1][1]*beta[1][0]

    #I_t = np.matmul(inv_alpha, beta)
    #I_d = I_t[0][0]
    #I_q = I_t[1][0]
    return I_d, I_q


def calculate_voltages(theta, I_d, I_q, Vs , theta_vs):
    """
    Calculate the voltage V_t based on the given inputs, for AVR model

    Parameters:
    theta (rad): The angle in radians.
    I_d (pu): The d-axis current.
    I_q (pu): The q-axis current.

    Returns:
    float: The magnitude of the total voltage V_t(pu).
    """
    Re = 0.1
    Xep = 0.1
    V_d = Re * I_d - Xep * I_q + Vs * np.sin(theta - theta_vs)
    V_q = Re * I_q + Xep * I_d + Vs * np.cos(theta - theta_vs)
    V_t = np.sqrt(V_d ** 2 + V_q ** 2)  # equal to Vs
    return V_t

