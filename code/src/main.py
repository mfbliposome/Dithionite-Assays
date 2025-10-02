import os
import joblib
from fluorescence_fit import *

def run_fluorescence_analysis(
    file_path, 
    sheet_name="Fluorescence 1_03", 
    results_folder='../../results/20251002',
    time_range =10,
    p0_no_triton=[25, 25, 0.001, 0.005]
):
    """
    Run full fluorescence analysis workflow for a single Excel file.
    Performs Triton analysis, then No-Triton (analytical and numerical) 
    with fixed k_deg_out from Triton results, and saves all results to a joblib file.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    sheet_name : str
        Name of the sheet containing fluorescence data.
    results_folder : str
        Folder to save results (joblib file and plots from modeling functions).
    time_range : float
        Time range to use for Triton analysis function.
    p0_no_triton : list
        Initial guess for No-Triton analytical fitting.

    Returns
    -------
    result_dict : dict
        Dictionary containing df_clean, results_triton, results_no_triton_fixed, results_no_triton_numerical
    """

    os.makedirs(results_folder, exist_ok=True)
    
    # Preprocess
    df_clean, filename = preprocess_data_auto(file_path, sheet_name)
    
    # Triton analysis
    results_triton = analyze_fluorescence_decay_triton(df_clean, filename, base_results_dir= results_folder, time_range=time_range)
    
    # Fixed k_deg_out for No-Triton analyses
    k_deg_out = results_triton['k_deg'].mean()
    
    # No Triton: analytical
    results_no_triton_fixed = analyze_fluorescence_decay_no_triton_fixed_kout(
        df_clean,
        filename,
        k_deg_out_fixed=k_deg_out,
        base_results_dir= results_folder,
        p0=p0_no_triton
    )
    
    # No Triton: numerical
    results_no_triton_numerical = analyze_fluorescence_decay_no_triton_numerical_fixed_kout(
        df_clean,
        filename,
        k_deg_out_fixed=k_deg_out,
        base_results_dir= results_folder,
    )
    
    # Store results
    result_dict = {
        'df_clean': df_clean,
        'results_triton': results_triton,
        'results_no_triton_fixed': results_no_triton_fixed,
        'results_no_triton_numerical': results_no_triton_numerical
    }
    
    # Save using joblib
    save_path = os.path.join(results_folder, f"{filename}_results.joblib")
    joblib.dump(result_dict, save_path)
    
    print(f"All results saved to {save_path}")
    return result_dict

# Usage example
# file_path = '../../data/PlateReader/20250718_sample2decylsulfate.xlsx'
# results = run_fluorescence_analysis(
#     file_path,
#     results_folder='../../results/my_analysis_folder'
# )