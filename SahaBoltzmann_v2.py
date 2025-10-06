# SahaBoltzmann_v2.py - Corrected Saha-Boltzmann plasma temperature estimation
# Fixed version of Refactored_SahaBoltzmann.py with all critical bugs addressed
# Based on analysis from corrected_saha_boltzmann.ipynb

import os
import json
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.constants import Boltzmann, speed_of_light, Planck, electron_mass, elementary_charge
from scipy.special import wofz
from scipy.stats import linregress
from readData import get_spectra

# -----------------------------
# File paths (modify as needed)
# -----------------------------
EXPERIMENT_FOLDER = "data\\REMUS-9951892\\FECRNI"
SAMPLE_LIST_PATH = "data\\Samples_Fe_matrix_2021_12_10.txt"
PARTITION_FUNCTION_PATH = "data\\PartF_var.txt"
EION_PATH = "data\\E_ion.txt"
QUANT_PAR_PATH = "data\\Quant_par_vacuum.txt"
LINE_DATA_PATH = "data\\Boltzmann_lines_v15.txt"

# -----------------------------
# Corrected Physical Constants
# -----------------------------
# FIXED: Use correct electron mass value (not electron_mass*1000)
kb = 1.380649e-16  # erg/K - Boltzmann constant
c = speed_of_light  # m/s - Speed of light  
h = 6.62607015e-27  # erg*s - Planck constant
me = 9.1093837e-28  # g - Correct electron mass (not electron_mass*1000)
e = elementary_charge  # C - Elementary charge

#------------------------------
# Voigt profile function
#------------------------------
def voigt_profile(wavelength, x0, sigma, gamma):
    """
    Calculate Voigt profile for spectral line shapes.
    """
    z = ((wavelength - x0) + 1j*gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# -----------------------------
# Partition function calculation
# -----------------------------
def partition_function(elem, T):
    """
    Calculate partition functions for neutral (I) and singly ionized (II) atoms.
    This function is correct in the original implementation.
    """
    df = pd.read_csv(PARTITION_FUNCTION_PATH, sep='\t', decimal='.')
    kb_eV = 8.617333262e-5  # eV/K

    df_I = df[(df['Element'] == elem) & (df['ionState'] == 'I')]
    df_II = df[(df['Element'] == elem) & (df['ionState'] == 'II')]

    U_I = np.sum(df_I['gi'] * np.exp(-df_I['Ei'] / (kb_eV * T)))
    U_II = np.sum(df_II['gi'] * np.exp(-df_II['Ei'] / (kb_eV * T)))
    
    U_I = float(U_I)
    U_II = float(U_II)

    return U_I, U_II

# -----------------------------
# Load experiment metadata
# -----------------------------
def load_metadata():
    """
    Load all required metadata files.
    """
    sample_list = pd.read_csv(SAMPLE_LIST_PATH, sep='\t', decimal='.')
    eion = pd.read_csv(EION_PATH, sep='\t', decimal='.')
    quant_par = pd.read_csv(QUANT_PAR_PATH, sep='\t', decimal='.')
    line_data = pd.read_csv(LINE_DATA_PATH, sep='\t', decimal='.')
    line_data.dropna(inplace=True)
    return sample_list, eion, quant_par, line_data

# -----------------------------
# Corrected Saha equation calculation
# -----------------------------
def calculate_saha_ratio(T, Ne, PF_I, PF_II, Eion_eV):
    """
    Calculate Saha ionization ratio with correct implementation.
    
    FIXED: Uses realistic electron density, correct exponent (3/2), and proper units
    """
    eV_to_erg = 1.60217e-12
    Eion_erg = Eion_eV * eV_to_erg
    
    # FIXED: Correct exponent (3/2) matching R implementation
    S10 = (((2*PF_II)/(Ne*PF_I)) * 
           ((me*kb*T)/((h**2)/(2*np.pi)))**(3/2) * 
           np.exp(-Eion_erg/(kb*T)))
    
    return S10

# -----------------------------
# Corrected Boltzmann plot construction
# -----------------------------
def construct_boltzmann_plots(intensities, wavelengths, gk, Ak, Ek, ion_states, 
                             T, Ne, PF_I, PF_II, S10, Eion, element_conc):
    """
    Construct separate Boltzmann plots for neutral and ionized lines.
    
    FIXED: Separates ionization states instead of mixing them incorrectly.
    """
    eV_to_erg = 1.60217e-12
    
    # FIXED: Separate neutral (I) and ionized (II) lines
    neutral_mask = (ion_states == 0)  # Ion state I
    ionized_mask = (ion_states == 1)  # Ion state II
    
    results = {}
    
    if np.any(neutral_mask):
        # Neutral lines (simple Boltzmann plot)
        I_neu = intensities[neutral_mask]
        wl_neu = wavelengths[neutral_mask] * 1e-9  # Convert nm to m
        gk_neu = gk[neutral_mask]
        Ak_neu = Ak[neutral_mask]
        Ek_neu = Ek[neutral_mask]
        
        # Standard Boltzmann plot: ln(I*λ/(g*A)) vs E_upper
        y_neutral = np.log((I_neu * wl_neu) / (gk_neu * Ak_neu))
        x_neutral = Ek_neu * eV_to_erg
        
        results['neutral'] = {'x': x_neutral, 'y': y_neutral, 'mask': neutral_mask}
    
    if np.any(ionized_mask):
        # Ionized lines (Saha-Boltzmann plot)
        I_ion = intensities[ionized_mask]
        wl_ion = wavelengths[ionized_mask] * 1e-9
        gk_ion = gk[ionized_mask]
        Ak_ion = Ak[ionized_mask]
        Ek_ion = Ek[ionized_mask]
        
        # FIXED: Proper Saha correction term with correct exponent
        saha_term = np.log(2 * (2*np.pi*me*kb*T)**(3/2) / (h**3 * Ne))
        
        # Saha-Boltzmann plot with proper corrections
        y_ionized = (np.log((I_ion * wl_ion) / (gk_ion * Ak_ion)) - 
                    saha_term - 
                    np.log(element_conc * S10 / (100 * PF_II)))
        
        # FIXED: Include ionization energy for II states
        x_ionized = (Ek_ion + Eion) * eV_to_erg
        
        results['ionized'] = {'x': x_ionized, 'y': y_ionized, 'mask': ionized_mask}
    
    return results

# -----------------------------
# Corrected temperature extraction
# -----------------------------
def extract_temperature_from_slope(x_data, y_data):
    """
    Extract temperature from Boltzmann plot slope.
    
    FIXED: Proper slope interpretation for temperature extraction.
    """
    if len(x_data) < 2:
        return None, None
    
    # Linear regression: y = mx + b, where m = -1/(k_B*T)
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    
    # FIXED: Correct temperature calculation from slope
    if slope < 0:  # Slope must be negative for physical validity
        T_fitted = -1 / (slope * kb)  # Temperature in Kelvin
        R2 = r_value**2
        return T_fitted, R2
    else:
        return None, None

# -----------------------------
# Corrected main Saha-Boltzmann analysis function
# -----------------------------
def saha_boltzmann_analysis(inp_data_idx, experimentList, sample_list, lineData, Eion, QparFinal):
    """
    Corrected Saha-Boltzmann analysis with all bugs fixed.
    """
    # Load spectral data
    json_path = os.path.join(EXPERIMENT_FOLDER, experimentList[inp_data_idx])
    w1, spectraData_1 = get_spectra(json_path, 1, 1, 1)
    w2, spectraData_2 = get_spectra(json_path, 1, 1, 2)
    wavelengths = np.concatenate([w1, w2])
    spectraAll = np.concatenate([spectraData_1, spectraData_2])

    # Sample metadata
    sample_name = experimentList[inp_data_idx].replace(".json", "").replace("/", "_")
    Cont = sample_list[sample_list['Sample Name'].str.replace("/", "_") == sample_name].copy()
    Cont = Cont.fillna(0)
    
    SahaBoltzmann_results = []
    
    # FIXED: Use realistic electron density (not placeholder Ne=1)
    Ne = 1.8e17  # cm^-3 - Realistic plasma electron density
    N = 1e-5  # Particle density parameter
    
    # Process elements
    elements = lineData['Elem_name'].unique()
    elements = elements[[5,6,7,0,1,2,3,4]]  # Reorder as in original
    
    print(f"Processing sample: {sample_name}")
    print(f"Using realistic electron density: Ne = {Ne:.2e} cm^-3")
    
    for elem in elements:
        # Get element-specific data
        Eion_temp = Eion[Eion['Element'] == f"{elem}+I"]
        lineData_temp = lineData[lineData['Elem_name'] == elem]
        Qpar_temp = QparFinal[QparFinal['Elem_name'] == elem]
        Cc = float(Cont[elem]) if elem in Cont.columns else 0
        
        if Cc == 0 or lineData_temp.empty or Eion_temp.empty:
            print(f"Skipping element {elem}: insufficient data or zero concentration")
            continue
            
        print(f"\nAnalyzing element: {elem}, Concentration: {Cc}%")
        
        # Extract line intensities with background correction
        data_lines = np.zeros((2, lineData_temp.shape[0]))
        line_info = []
        
        for u in range(lineData_temp.shape[0]):
            # FIXED: Proper array indexing for wavelength regions
            try:
                selection = ((wavelengths > lineData_temp.iloc[u, 8]) & 
                           (wavelengths < lineData_temp.iloc[u, 9]))
                bckgrnd = ((wavelengths > lineData_temp.iloc[u, 10]) & 
                          (wavelengths < lineData_temp.iloc[u, 11]))
                
                if not np.any(selection) or not np.any(bckgrnd):
                    continue
                    
                background_average = np.mean(spectraAll[bckgrnd])
                data_sum = abs(np.sum(spectraAll[selection])) - (np.sum(selection) * background_average)
                data_max = np.max(spectraAll[selection]) - background_average
                
                if data_max <= 0:  # Skip negative or zero intensities
                    continue
                    
                data_lines[0, u] = data_sum
                data_lines[1, u] = data_max
                
                # Store line information
                line_info.append({
                    'wavelength': lineData_temp.iloc[u, 2],  # Wavelength in nm
                    'gk': lineData_temp.iloc[u, 7],  # Upper level statistical weight  
                    'Ak': lineData_temp.iloc[u, 5],  # Transition probability
                    'Ek': lineData_temp.iloc[u, 4],  # Upper level energy in eV
                    'ion_state': 1 if lineData_temp.iloc[u, 1] == "II" else 0,  # Ionization state
                    'intensity': data_max,
                    'index': u
                })
                
            except (IndexError, ValueError) as e:
                print(f"Error processing line {u} for element {elem}: {e}")
                continue
        
        if not line_info:
            print(f"No valid lines found for element {elem}")
            continue
            
        # Convert line info to arrays
        valid_indices = [info['index'] for info in line_info]
        intensities = np.array([info['intensity'] for info in line_info])
        wavelengths_nm = np.array([info['wavelength'] for info in line_info])
        gk_array = np.array([info['gk'] for info in line_info])
        Ak_array = np.array([info['Ak'] for info in line_info])
        Ek_array = np.array([info['Ek'] for info in line_info])
        ion_states = np.array([info['ion_state'] for info in line_info])
        
        # Iterative temperature determination
        T_current = 10000  # Initial temperature guess (K)
        max_iterations = 20
        tolerance = 50  # K
        
        print(f"Starting temperature iteration with {len(line_info)} valid lines")
        print(f"Ion states: {np.sum(ion_states == 0)} neutral, {np.sum(ion_states == 1)} ionized")
        
        for iteration in range(max_iterations):
            # Calculate partition functions at current temperature
            PF_I, PF_II = partition_function(elem, T_current)
            
            # Calculate Saha ratio with corrected implementation
            Eion_eV = Eion_temp['Eion'].values[0] if not Eion_temp.empty else 7.9
            S10 = calculate_saha_ratio(T_current, Ne, PF_I, PF_II, Eion_eV)
            
            # FIXED: Construct separate Boltzmann plots by ionization state
            boltz_plots = construct_boltzmann_plots(
                intensities, wavelengths_nm, gk_array, Ak_array, Ek_array, 
                ion_states, T_current, Ne, PF_I, PF_II, S10, Eion_eV, Cc
            )
            
            # Extract temperatures from each ionization state
            temperatures = []
            r2_values = []
            
            for state_name, plot_data in boltz_plots.items():
                T_fitted, R2 = extract_temperature_from_slope(plot_data['x'], plot_data['y'])
                if T_fitted is not None and 5000 < T_fitted < 25000:  # Reasonable temperature range
                    temperatures.append(T_fitted)
                    r2_values.append(R2)
                    print(f"  {state_name} lines: T = {T_fitted:.0f} K, R² = {R2:.3f}")
            
            if not temperatures:
                print(f"  No valid temperature fits in iteration {iteration}")
                break
                
            # FIXED: Average temperatures from different ionization states
            T_new = np.mean(temperatures)
            avg_R2 = np.mean(r2_values)
            
            dT = abs(T_new - T_current)
            print(f"  Iteration {iteration}: T = {T_new:.0f} K, ΔT = {dT:.0f} K, avg R² = {avg_R2:.3f}")
            
            # Check convergence
            if dT < tolerance:
                print(f"  ✓ Converged: Final T = {T_new:.0f} K")
                
                # Store results
                result = {
                    "Sample": experimentList[inp_data_idx],
                    "Element": elem,
                    "Temp": T_new,
                    "R2": avg_R2,
                    "U_I": PF_I,
                    "U_II": PF_II,
                    "S10": S10,
                    "Ne": Ne,
                    "Concentration": Cc,
                    "NumLines": len(line_info),
                    "NumNeutral": np.sum(ion_states == 0),
                    "NumIonized": np.sum(ion_states == 1),
                    "Converged": True,
                    "Iterations": iteration + 1
                }
                SahaBoltzmann_results.append(result)
                break
                
            T_current = T_new
            
            # Avoid infinite loops
            if iteration >= max_iterations - 1:
                print(f"  ⚠ Did not converge after {max_iterations} iterations")
                result = {
                    "Sample": experimentList[inp_data_idx],
                    "Element": elem,
                    "Temp": T_current,
                    "R2": avg_R2 if r2_values else 0,
                    "U_I": PF_I,
                    "U_II": PF_II,
                    "S10": S10,
                    "Ne": Ne,
                    "Concentration": Cc,
                    "NumLines": len(line_info),
                    "NumNeutral": np.sum(ion_states == 0),
                    "NumIonized": np.sum(ion_states == 1),
                    "Converged": False,
                    "Iterations": max_iterations
                }
                SahaBoltzmann_results.append(result)
    
    # Convert results to DataFrame
    if SahaBoltzmann_results:
        results_df = pd.DataFrame(SahaBoltzmann_results)
        
        # FIXED: Apply quality filters similar to original
        quality_filtered = results_df[
            (results_df['Temp'] < 16000) & 
            (results_df['Temp'] > 8800) & 
            (results_df['R2'] > 0.8) &  # Slightly relaxed R² threshold
            (results_df['Converged'] == True)
        ]
        
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total elements processed: {len(results_df)}")
        print(f"Quality-filtered results: {len(quality_filtered)}")
        
        return quality_filtered
    else:
        print("No valid results obtained")
        return pd.DataFrame()

# -----------------------------
# Main execution block
# -----------------------------
def main():
    """
    Main function with improved error handling and reporting.
    """
    print("=== Saha-Boltzmann Temperature Analysis v2.0 ===")
    print("Corrected implementation with all critical bugs fixed")
    print()
    
    inp_data_idx = 14  # Index of experiment to process
    
    try:
        # Load metadata
        print("Loading metadata...")
        sample_list, Eion, QparFinal, lineData = load_metadata()
        print(f"✓ Loaded data: {len(sample_list)} samples, {len(lineData)} lines")
        
        # Get experiment list
        if not os.path.exists(EXPERIMENT_FOLDER):
            print(f"✗ Experiment folder not found: {EXPERIMENT_FOLDER}")
            print("Please update EXPERIMENT_FOLDER path")
            return
            
        experimentList = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.json')]
        
        if not experimentList:
            print(f"✗ No JSON files found in {EXPERIMENT_FOLDER}")
            return
            
        if inp_data_idx >= len(experimentList):
            print(f"✗ Index {inp_data_idx} exceeds available experiments ({len(experimentList)})")
            return
            
        print(f"✓ Found {len(experimentList)} experiment files")
        print(f"Processing experiment: {experimentList[inp_data_idx]}")
        
        # Run analysis
        results = saha_boltzmann_analysis(inp_data_idx, experimentList, sample_list, 
                                        lineData, Eion, QparFinal)
        
        if not results.empty:
            print(f"\n=== FINAL RESULTS ===")
            print("Element   Temperature(K)   R²      Lines  Status")
            print("-" * 50)
            for _, row in results.iterrows():
                status = "✓ Good" if row['Converged'] and row['R2'] > 0.9 else "⚠ Check"
                print(f"{row['Element']:<8} {row['Temp']:>8.0f}     {row['R2']:>5.3f}   "
                      f"{row['NumLines']:>3}    {status}")
                      
            # Calculate average temperature
            avg_temp = results['Temp'].mean()
            temp_std = results['Temp'].std()
            print(f"\nAverage plasma temperature: {avg_temp:.0f} ± {temp_std:.0f} K")
            
            return results
        else:
            print("\n✗ No quality results obtained. Check:")
            print("  - Data file paths and formats")
            print("  - Spectral line quality (S/N ratio)")
            print("  - Element concentrations")
            print("  - Wavelength calibration")
            
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("Please check data file paths in the configuration section")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        print("Check input data format and experimental conditions")

if __name__ == '__main__':
    results = main()