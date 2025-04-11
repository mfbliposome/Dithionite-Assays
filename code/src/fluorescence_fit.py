import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error

from datetime import datetime


def preprocess_data(file_path):
    '''
    Preprocess the data file and make it ready for analysis
    '''

    dfs = pd.read_csv(file_path)
    filename_stem = os.path.splitext(os.path.basename(file_path))[0]

    # Define the columns to keep (Notice, there is a space after Trial 2)
    # columns_to_keep = ["Unnamed: 0", "31 Trial 1", "31 Trial 2 ", "31 Trial 3", 
    #                 "31 Triton Trial 1", "31 Triton Trial 2 ", "31 Triton Trial 3"]

    # Keep only the selected columns
    df_filtered = dfs[:]

    # Rename 'Unnamed: 0' to 'time' for clarity
    df_filtered = df_filtered.rename(columns={"Unnamed: 0": "time"})
    df_filtered = df_filtered.iloc[1:].reset_index(drop=True)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')

    return df_filtered, filename_stem


def analyze_fluorescence_decay_triton(data, filename, time_range=60):
    '''
    Analyze fluorescence decay following Triton treatment using a linearized 
    exponential decay model and save the fitted parameters and plots.

    This function fits a linear model to the natural logarithm of fluorescence 
    intensity data over time to estimate the initial fluorescence (F0) and 
    the degradation rate constant (k_deg) for each of the last three columns 
    in the input data (representing three Triton-treated trials). It also computes
    R² and RMSE to evaluate fit quality and generates a residual plot.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing time in the first column and fluorescence 
        intensity values in the last three columns.

    filename : str
        The base filename (without extension) to use for saving the plot and CSV output.
        The base filename will obtained after preprocess original data file

    time_range : int, optional
        Number of time steps used for fitting (default is 60).

    Returns
    -------
    param_df : pandas.DataFrame
        A DataFrame containing the fitted values of F0, k_deg, R², and RMSE 
        for each trial. The index corresponds to the trial numbers.
    '''

    # Extract time and fluorescence data
    time_total = data.iloc[:, 0].values
    F_trials_total = data.iloc[:, -3:].values

    time = data.iloc[:time_range, 0].values
    F_trials = data.iloc[:time_range, -3:].values
    ln_F_trials = np.log(F_trials)

    # Define the linear model for fitting
    def linear_model(t, ln_F0, k_deg):
        return ln_F0 - k_deg * t
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H:%M')
    results_dir = f'../../results/{filename}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize plot for fit
    plt.figure(figsize=(10, 6))
    
    # Store fit parameters
    params = []
    params_ori = []

    for i in range(3):
        ln_F = ln_F_trials[:, i]
        initial_guess = [ln_F[0], 0.01]

        # Fit the model
        popt, _ = curve_fit(linear_model, time, ln_F, p0=initial_guess)
        ln_F0, k_deg = popt
        F0 = np.exp(ln_F0)
        params.append([F0, k_deg])
        params_ori.append(popt)

        # Plot data and fit
        plt.scatter(time_total, np.log(F_trials_total)[:, i], s=8, label=f'Triton Trial {i+1} Data', alpha=0.5)
        plt.plot(time, linear_model(time, *popt), '--', label=f'Fit {i+1}, k={k_deg:.4f}', linewidth=2)

    # Finalize and save fit plot
    plt.xlabel("Time")
    plt.ylabel("ln(Fluorescence Intensity)")
    plt.title("Fluorescence Decay with Triton (Log Scale)")
    plt.legend()
    fit_plot_filename = f"{filename}_fit_plot_Triton.png"
    plot_path = os.path.join(results_dir, fit_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals + metrics plot
    plt.figure(figsize=(10, 4))
    r2s = []
    rmses = []

    for i in range(3):
        ln_F = ln_F_trials[:, i]
        ln_F_fit = linear_model(time, *params_ori[i])
        residuals = ln_F - ln_F_fit
        plt.scatter(time, residuals, s=8, label=f'Triton Trial {i+1}', alpha=0.7)

        r2 = r2_score(ln_F, ln_F_fit)
        rmse = np.sqrt(mean_squared_error(ln_F, ln_F_fit))
        r2s.append(r2)
        rmses.append(rmse)
        print(f"Trial {i+1}: R^2 = {r2:.4f}, RMSE = {rmse:.4f}")

    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residuals (ln(Fluorescence) - Fit)")
    plt.title("Residual Plot (With Triton)")
    plt.grid(True)
    plt.legend()
    residual_plot_filename = f"{filename}_residual_plot_Triton.png"
    plot_path = os.path.join(results_dir, residual_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Combine all results into DataFrame
    param_df = pd.DataFrame(
        [(F0, k, r2s[i], rmses[i]) for i, (F0, k) in enumerate(params)],
        columns=["F0", "k_deg", "R_squared", "RMSE"],
        index=[f"Trial {i+1}" for i in range(3)]
    )

    csv_filename = f"{filename}_fit_params_Triton.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    param_df.to_csv(csv_path, index=False)

    print(f"Saved: {csv_filename}")
    print(f"Saved: {fit_plot_filename}")
    print(f"Saved: {residual_plot_filename}")
    return param_df

def analyze_fluorescence_decay_no_triton(data, filename, p0 =[25, 25, 0.01, 0.001, 0.005]):
    """
    Fit fluorescence decay curves (no Triton) using a composite exponential model 
    and save fit results, plots, and statistics.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a 'time' column and three fluorescence columns.

    Initial guess p0: list
        Initial guess for curve fitting parameters [C1, C2, k_deg_out, k_deg_in, k_perm].

    filename : str
        Base filename for saving plot and CSV outputs (no extension).

    Returns
    -------
    param_df : pandas.DataFrame
        DataFrame containing fitted parameters, R², and RMSE for each trial.
    """

    t = data['time'].values
    columns = data.columns[1:4]
    colors = ['blue', 'orange', 'green']
    results = []

    def F_total(t, C1, C2, k_deg_out, k_deg_in, k_perm):
        term_in = np.exp(-k_deg_in * (t + np.exp(-k_perm * t) / k_perm))
        term_out = np.exp(-k_deg_out * t)
        return C1 * term_out + C2 * term_in
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H:%M')
    results_dir = f'../../results/{filename}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # --- Fit Plot ---
    plt.figure(figsize=(10, 6))

    for i, col in enumerate(columns):
        F = data[col].values
        F0_guess = F[0]
        # p0 = [F0_guess / 2, F0_guess / 2, 0.01, 0.001, 0.005]  # Initial guess
        
        # Update bounds based on F0
        lb = [0, 0, 1e-10, 1e-10, 1e-10]
        ub = [F0_guess, F0_guess, 10.0, 10.0, 10.0]

        try:
            popt, _ = curve_fit(F_total, t, F, p0=p0, bounds=(lb, ub), maxfev=50000)
            F_fit = F_total(t, *popt)

            # Compute metrics
            r2 = r2_score(F, F_fit)
            rmse = np.sqrt(mean_squared_error(F, F_fit))

            results.append((col, *popt, r2, rmse))

            # Plot
            plt.scatter(t, F, label=f"{col} (data)", color=colors[i], alpha=0.3)
            plt.plot(t, F_fit, label=f"{col} (fit)", color=colors[i], linestyle='-')
        except RuntimeError:
            print(f"Fit failed for {col}")
            results.append((col, *[np.nan]*5, np.nan, np.nan))

    plt.xlabel('Time')
    plt.ylabel('Fluorescence')
    plt.title('Fluorescence Decay Fit (No Triton)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fit_plot_filename = f"{filename}_fit_plot_noTriton.png"
    plot_path = os.path.join(results_dir, fit_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Residual Plot ---
    plt.figure(figsize=(10, 4))

    for i, (col, *params) in enumerate(results):
        if np.isnan(params[0]):
            continue
        F = data[col].values
        F_fit = F_total(t, *params[:5])
        residuals = F - F_fit
        plt.scatter(t, residuals, s=10, label=f'{col} residuals', alpha=0.7)

    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (No Triton)")
    plt.legend()
    plt.grid(True)
    residual_plot_filename = f"{filename}_residual_plot_noTriton.png"
    plot_path = os.path.join(results_dir, residual_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save to CSV ---
    param_df = pd.DataFrame(results, columns=[
        "Trial", "F_out0", "F_in0", "k_deg_out", "k_deg_in", "k_perm", "R_squared", "RMSE"
    ])
    csv_filename = f"{filename}_fit_params_noTriton.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    param_df.to_csv(csv_path, index=False)

    print(f"Saved fit plot to: {fit_plot_filename}")
    print(f"Saved residual plot to: {residual_plot_filename}")
    print(f"Saved parameters to: {csv_filename}")
    
    return param_df


def analyze_fluorescence_decay_no_triton_numerical(data, filename):
    """
    Analyze fluorescence decay data using numerical solution.
    Saves fit plots and parameters (including R² and RMSE) to a timestamped results folder.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with time in the first column and at least three fluorescence trials.

    Returns
    -------
    param_df : pandas.DataFrame
        DataFrame with fitted parameters and statistics for each trial.
    """

    # Setup
    time = data.iloc[:, 0].values
    fluorescence_trials = [data.iloc[:, i].values for i in range(1, 4)]
    results=[]

    # Define ODE model
    def fluorescence_model(t, F_out0, F_in0, k_deg_out, k_deg_in, k_perm):
        def odes(t, F):
            F_out, F_in = F
            dF_out_dt = -k_deg_out * F_out
            dF_in_dt = -k_deg_in * F_in * (1 - np.exp(-k_perm * t))
            return [dF_out_dt, dF_in_dt]
        
        sol = solve_ivp(odes, [t[0], t[-1]], [F_out0, F_in0], t_eval=t, method='RK45')
        return sol.y[0] + sol.y[1]

    def fit_function(t, F_out0, F_in0, k_deg_out, k_deg_in, k_perm):
        return fluorescence_model(t, F_out0, F_in0, k_deg_out, k_deg_in, k_perm)

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H:%M')
    results_dir = f'../../results/{filename}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Fit and plot
    params_list = []
    fit_curves = []
    residuals_list = []
    colors = plt.cm.tab10.colors

    for i, fluorescence in enumerate(fluorescence_trials):
        initial_guess = [fluorescence[0] * 0.5, fluorescence[0] * 0.5, 0.01, 0.01, 0.001]

        try:
            popt, _ = curve_fit(fit_function, time, fluorescence, p0=initial_guess, maxfev=10000)
            F_fit = fit_function(time, *popt)

            # Calculate stats
            residuals = fluorescence - F_fit
            r2 = r2_score(fluorescence, F_fit)
            rmse = np.sqrt(mean_squared_error(fluorescence, F_fit))

            params_list.append((f"Trial_{i+1}", *popt, r2, rmse))
            fit_curves.append(F_fit)
            residuals_list.append(residuals)

           

        except RuntimeError:
            print(f"Fit failed for Trial {i+1}")
            params_list.append((f"Trial_{i+1}", *[np.nan]*5, np.nan, np.nan))
            fit_curves.append(np.full_like(time, np.nan))
            residuals_list.append(np.full_like(time, np.nan))

    # Plot: Fit
    plt.figure(figsize=(10, 6))
    for i, (fluorescence, F_fit) in enumerate(zip(fluorescence_trials, fit_curves)):
        plt.scatter(time, fluorescence, color=colors[i], label=f'Trial {i+1} Data', alpha=0.3)
        plt.plot(time, F_fit, '-', color=colors[i], label=f'Trial {i+1} Fit')
    plt.xlabel("Time")
    plt.ylabel("Fluorescence")
    plt.title("Fluorescence Decay Fit (No Triton, Numerical)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fit_plot_filename = f"{filename}_fit_plot_noTriton(Numerical).png"
    plot_path = os.path.join(results_dir, fit_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot: Residuals
    plt.figure(figsize=(10, 4))
    for i, residuals in enumerate(residuals_list):
        plt.scatter(time, residuals, s=10, color=colors[i], label=f'Trial {i+1}', alpha=0.7)
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.title("Residual Plot (No Triton, Numerical)")
    plt.legend()  
    plt.grid(True)
    plt.tight_layout()
    residual_plot_filename = f"{filename}_residual_plot_noTriton(Numerical).png"
    plot_path = os.path.join(results_dir, residual_plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save parameters to CSV
    param_df = pd.DataFrame(params_list, columns=[
        "Trial", "F_out0", "F_in0", "k_deg_out", "k_deg_in", "k_perm", "R_squared", "RMSE"
    ])
    csv_filename = f"{filename}_fit_params_noTriton(Numerical).csv"
    csv_path = os.path.join(results_dir, csv_filename)
    param_df.to_csv(csv_path, index=False)

    print(f"Results saved to {results_dir}")
    return param_df



