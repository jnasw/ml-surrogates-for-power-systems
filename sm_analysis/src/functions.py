import numpy as np
import matplotlib.pyplot as plt
import inspect
import torch

# Set time
def set_time(end_time, interval_points):
    t_span = (0, end_time)
    #step = end_time/interval_points
    #t_eval = np.arange(0, end_time+step, step)
    t_eval = np.linspace(0, end_time, interval_points + 1) # we want to include the starting point and the end point
    return t_span, t_eval

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set in hydra
    np.random.seed(random_seed)# done
    #random.seed(random_seed) # done   
    return   

# Plotting the solution
def plotting_solution(solution,flag=True):
    plt.figure(figsize=(10, 6))
    if flag:
        plt.plot(solution.t, solution.y[0], label='δ')
    plt.plot(solution.t, solution.y[1], label='ω (rad/s)')
    plt.title('Synchronous Machine - Infinite Bus')
    plt.xlabel('Time (s)')
    plt.ylabel('Values (pu)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the solution
def interactive_plotting_solution(solution,flag=True):
    plt.figure(figsize=(10, 6))
    if flag:
        plt.plot(solution.t, solution.y[0], label='δ')
    plt.plot(solution.t, solution.y[1], label='ω (rad/s)')
    plt.title('Synchronous Machine - Infinite Bus')
    plt.xlabel('Time (s)')
    plt.ylabel('Values (pu)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sol(sol):
    plt.figure(figsize=(10, 6))
    for i in range(len(sol)):
        solution = sol[i]
        plt.plot(solution.t, solution.y[1], label='δ')
        #plt.plot(solution.t, solution.y[1], label='Omega (pu)')
    plt.title('Synchronous Machine - Infinite Bus')
    plt.xlabel('Time (s)')
    plt.ylabel('Values (pu)')
    plt.show()
    return

# Plotting the solution with gridspec
def plotting_solution_gridspec_original(sol, model, show=True):
    fig, axs = plt.subplots(4, 1, figsize=(8, 8)) 
    # check if sol[0] exists

    if 'message' in sol:
        solution = sol
        axs[0].plot(solution.t, solution.y[0], label='δ')
        axs[1].plot(solution.t, solution.y[1], label='ω (rad/s)')
        axs[2].plot(solution.t, solution.y[2], label='E_d_dash (pu)')
        axs[3].plot(solution.t, solution.y[3], label='E_q_dash (pu)')
    else:
        for i in range(len(sol)):
            solution = sol[i]
            axs[0].plot(solution.t, solution.y[0], label='δ')
            axs[1].plot(solution.t, solution.y[1], label='ω (rad/s)')
            axs[2].plot(solution.t, solution.y[2], label='E_d_dash (pu)')
            axs[3].plot(solution.t, solution.y[3], label='E_q_dash (pu)')
    
    axs[0].set_title("Machine number "+str(model))
    axs[0].set_ylabel('δ')
    axs[0].grid(True)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('ω (rad/s)')
    axs[1].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('E_d_dash (pu)')
    axs[2].grid(True)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('E_q_dash (pu)')
    axs[3].grid(True)
    if show:
        plt.show()
    return None

# Plotting the solution with gridspec
def plotting_solution_gridspec_original_all(sol, model, show=True):
    

    # check if sol[0] exists
    
    if 'message' in sol:
        solution = sol
        
        if len(solution.y)>7:
            height=16
            fig, axs = plt.subplots(9, 1, figsize=(8, height))
        else:
            if len(sol.y)>4:
                height=12
                fig, axs = plt.subplots(7, 1, figsize=(8, height))
            else:
                height=8
                fig, axs = plt.subplots(4, 1, figsize=(8, height))
        axs[0].plot(solution.t, solution.y[0], label='δ')
        axs[1].plot(solution.t, solution.y[1], label='ω (rad/s)')
        axs[2].plot(solution.t, solution.y[2], label='E_d_dash (pu)')
        axs[3].plot(solution.t, solution.y[3], label='E_q_dash (pu)')
        if len(solution.y)>4:
            axs[4].plot(solution.t, solution.y[4], label='R_f (pu)')
            axs[5].plot(solution.t, solution.y[5], label='V_R (pu)')
            axs[6].plot(solution.t, solution.y[6], label='E_fd (pu)')
        if len(solution.y)>7:
            axs[7].plot(solution.t, solution.y[7], label='P_m (pu)')
            axs[8].plot(solution.t, solution.y[8], label='P_sv (pu)')

    else:
        for i in range(len(sol)):
            solution = sol[i]
            if len(solution.y)>7:
                height=16
                fig, axs = plt.subplots(9, 1, figsize=(8, height))
            else:
                if len(solution.y)>4:
                    height=12
                    fig, axs = plt.subplots(7, 1, figsize=(8, height))
                else:
                    height=8
                    fig, axs = plt.subplots(4, 1, figsize=(8, height))
            axs[0].plot(solution.t, solution.y[0], label='δ')
            axs[1].plot(solution.t, solution.y[1], label='ω (rad/s)')
            axs[2].plot(solution.t, solution.y[2], label='E_d_dash (pu)')
            axs[3].plot(solution.t, solution.y[3], label='E_q_dash (pu)')
            if len(solution.y)>4:
                axs[4].plot(solution.t, solution.y[4], label='R_f (pu)')
                axs[5].plot(solution.t, solution.y[5], label='V_R (pu)')
                axs[6].plot(solution.t, solution.y[6], label='E_fd (pu)')
            if len(solution.y)>7:
                axs[7].plot(solution.t, solution.y[7], label='P_m (pu)')
                axs[8].plot(solution.t, solution.y[8], label='P_sv (pu)')
    
    axs[0].set_title("Machine number "+str(model))
    axs[0].set_ylabel('δ')
    axs[0].grid(True)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('ω (rad/s)')
    axs[1].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('E_d_dash (pu)')
    axs[2].grid(True)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('E_q_dash (pu)')
    axs[3].grid(True)
    if len(solution.y)>4:
        axs[4].set_xlabel('Time (s)')
        axs[4].set_ylabel('R_f (pu)')
        axs[4].grid(True)
        axs[5].set_xlabel('Time (s)')
        axs[5].set_ylabel('V_R (pu)')
        axs[5].grid(True)
        axs[6].set_xlabel('Time (s)')
        axs[6].set_ylabel('E_fd (pu)')
        axs[6].grid(True)
    if len(solution.y)>7:
        axs[7].set_xlabel('Time (s)')
        axs[7].set_ylabel('P_m (pu)')
        axs[7].grid(True)
        axs[8].set_xlabel('Time (s)')
        axs[8].set_ylabel('P_sv (pu)')
        axs[8].grid(True)
    if show:
        plt.show()
    return None

# Find missing parameters
def find_missing_params(func, parameters_):
    for i in inspect.getfullargspec(func)[0]:
        if i not in parameters_ and i not in ["x", "y","t","theta","omega","E_d_dash","E_q_dash"]:
            print(i)

def checkflag(not_ib_flag,avr_flag,gov_flag):
    #check if more than twp flags are true
    if not_ib_flag and avr_flag:
        print("Please select only one model")
        return False
    if not_ib_flag and gov_flag:
        print("Please select only one model")
        return False
    if avr_flag and gov_flag:
        print("Please select only one model")
        return False
    return True


def calculate_current(theta, E_d_dash, E_q_dash, X_d_dash, X_q_dash, Rs, Vs, theta_vs):
    Rs=0.0
    Re=0.0
    Xep=0.0
    alpha = [[(Rs+Re), -(X_q_dash+Xep)], [(X_d_dash+Xep), (Rs+Re)]]
    beta = [[E_d_dash - Vs*np.sin(theta-theta_vs)], [E_q_dash - Vs*np.cos(theta-theta_vs)]]

    inv_alpha = np.linalg.inv(alpha)
    I_d= inv_alpha[0][0]*beta[0][0] + inv_alpha[0][1]*beta[1][0]
    I_q= inv_alpha[1][0]*beta[0][0] + inv_alpha[1][1]*beta[1][0]
        
    #I_t = np.matmul(inv_alpha, beta)
    #I_d = I_t[0][0]
    #I_q = I_t[1][0]
    return I_d, I_q


def log_data_metrics_to_wandb(run, config):
    run.log({"Total time of simulation ": config.time})
    run.log({"Number of samples ": config.num_of_points})
    run.log({"Test with model ": config.model.model_flag})
    run.log({"Machine number ": config.model.machine_num })
    run.log({"Initial conditions ": config.model.init_condition_bounds})
    run.log({"Sampling method for initial points ": config.model.sampling})
    return 

def log_pinn_metrics_to_wandb(run,config):
    run.log({"Number of hidden layers ": config.nn.hidden_layers})
    run.log({"Number of hidden dimensions ": config.nn.hidden_dim})
    run.log({"Loss criterion ": config.nn.loss_criterion})
    run.log({"Optimizer ": config.nn.optimizer})
    run.log({"Weight initialization ": config.nn.weight_init})
    run.log({"Learning rate ": config.nn.lr})
    run.log({"Learning rate scheduler ": config.nn.lr_scheduler})
    run.log({"Number of epochs ": config.nn.num_epochs})
    run.log({"Batch size ": config.nn.batch_size})
    run.log({"Shuffle ": config.dataset.shuffle})

    return

def log_losses_and_weights_to_wandb(run, epoch, loss_data, loss_dt, loss_pinn, loss_total, weight_data, weight_dt, weight_pinn):
    if run is not None:
        run.log({"Loss_data": loss_data.item(), 'epoch': epoch})
        run.log({"Loss_dt": loss_dt, 'epoch': epoch})
        run.log({"Loss_pinn": loss_pinn, 'epoch': epoch})
        run.log({"Loss_total": loss_total.item(), 'epoch': epoch})
        run.log({"Weight_data": weight_data, 'epoch': epoch})
        run.log({"Weight_dt": weight_dt, 'epoch': epoch})
        run.log({"Weight_pinn": weight_pinn, 'epoch': epoch})
    return

def plotting_solution_gridspec(sol, modelling, model, num_of_points):
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    for i in range(len(sol)):
        solution = sol[i]
        axs[0].plot(solution[0][:num_of_points], solution[1][:num_of_points], label='Theta (pu)')
        axs[1].plot(solution[0][:num_of_points], solution[2][:num_of_points], label='Omega (pu)')
        axs[2].plot(solution[0][:num_of_points], solution[3][:num_of_points], label='E_d_dash (pu)')
        axs[3].plot(solution[0][:num_of_points], solution[4][:num_of_points], label='E_q_dash (pu)')
    
    axs[0].set_title(modelling+" for model num :"+str(model))
    axs[0].set_ylabel('δ')
    axs[0].grid(True)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('ω (rad/s)')
    axs[1].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('E_d_dash')
    axs[2].grid(True)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('E_q_dash')
    axs[3].grid(True)
    plt.show()
    return None

def plotting_solution_gridspec_dt(network, sol, modelling, model, num_of_points):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    for i in range(len(sol)):
        solution = sol[i]
        solution = torch.tensor(solution, dtype=torch.float32).T
        time = solution[:num_of_points,0]
        solution = solution[:num_of_points,1:]
        dt=network.calculate_from_ode(solution)
        dt= dt.to('cpu').detach().numpy()
        axs[0].plot(time, dt[:,0] , label='δ')
        axs[1].plot(time, dt[:,1] , label='ω (rad/s)')
        axs[2].plot(time, dt[:,2] , label='E_d_dash (pu)')
        axs[3].plot(time, dt[:,3] , label='E_q_dash (pu)')
     #+" for Machine model number :"+str(model)
    axs[0].set_title(modelling)
    axs[1].set_xlabel('Time (s)')
    axs[0].set_ylabel('δ')
    axs[0].grid(True)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('ω (rad/s)')
    axs[1].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('E_d_dash')
    axs[2].grid(True)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('E_q_dash')
    axs[3].grid(True)
    plt.show()
    return None


def plotting_target_gridspec_dt(network, target, time, modelling, model, num_of_points):
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    solution = torch.tensor(solution, dtype=torch.float32).T
    solution = solution[:num_of_points,1:]
    dt=network.calculate_from_ode(target)
    dt= dt.to('cpu').detach().numpy()
    axs[0].plot(time, dt[:,0] , label='δ')
    axs[1].plot(time, dt[:,1] , label='ω (rad/s)')
    axs[2].plot(time, dt[:,2] , label='E_d_dash (pu)')
    axs[3].plot(time, dt[:,3] , label='E_q_dash (pu)')
    
    axs[0].set_title(modelling+" for model num :"+str(model))
    axs[1].set_xlabel('Time (s)')
    axs[0].set_ylabel('dδ')
    axs[0].grid(True)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('dω')
    axs[1].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('DE_d_dash')
    axs[2].grid(True)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('DE_q_dash')
    axs[3].grid(True)
    plt.show()
    return None