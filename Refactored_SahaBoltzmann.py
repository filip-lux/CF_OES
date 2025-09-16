
# Refactored Python script for Saha-Boltzmann plasma temperature estimation
# Translated from MultiElemSBPlot.txt

import os
import json
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.constants import Boltzmann, speed_of_light, Planck, electron_mass, elementary_charge

# -----------------------------
# File paths (modify as needed)
# -----------------------------
EXPERIMENT_FOLDER = "P:/Running_projects/24_0011_CF_spark/CF SPARK/Data/Chameleon_OptiCal_DB_Calibration_Data_Fe_20211126/REMUS-9951892/FECRNI/"
SAMPLE_LIST_PATH = "D:/ownCloud/Documents/Granty/2023/NCK Thermofisher 2023/Data/Samples_Fe_matrix_2021_12_10.txt"
PARTITION_FUNCTION_PATH = "d:/Gdrive/Projects/LIBS/Spectra modelling/R/PartF_var.txt"
EION_PATH = "d:/Gdrive/Projects/LIBS/Spectra modelling/R/E_ion.txt"
QUANT_PAR_PATH = "D:/ownCloud/Documents/Projects/NCK_ThermoFisher/CFSparkOES/Quant_par_vacuum.txt"
LINE_DATA_PATH = "P:/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Code/R/Boltzmann_lines_v15.txt"

# -----------------------------
# Constants
# -----------------------------
kb = Boltzmann  # J/K
c = speed_of_light  # m/s
h = Planck  # J.s
me = electron_mass  # kg
e = elementary_charge  # C

# -----------------------------
# Function to read JSON spectral data
# -----------------------------
def read_spark_data(json_path, runID, integrationPhase, wvlRange):
    with open(json_path, 'r') as f:
        j = json.load(f)['analysis']

    spectra_data = j['results']['spectraData'][str(runID)]['spectra'][str(integrationPhase)]
    ccd_range_intensities = spectra_data['results'][str(wvlRange)]
    p2w = list(reversed(spectra_data['pixelToWaveLength'][str(wvlRange)]))
    drift = spectra_data['drift']['beta'][str(wvlRange)] + spectra_data['drift']['alpha'][str(wvlRange)]

    drift_cor = Polynomial(drift)
    pol = Polynomial(p2w)
    wvl = drift_cor(np.arange(1, len(ccd_range_intensities) + 1))
    wavelengths = pol(wvl)

    return pd.DataFrame({"wavelengths": wavelengths, "intensities": ccd_range_intensities})

# -----------------------------
# Function to calculate partition function
# -----------------------------
def partition_function(elem, T):
    df = pd.read_csv(PARTITION_FUNCTION_PATH, sep='	', decimal='.')
    kb_eV = 8.617333262e-5  # eV/K

    df_I = df[(df['Element'] == elem) & (df['ionState'] == 'I')]
    df_II = df[(df['Element'] == elem) & (df['ionState'] == 'II')]

    U_I = np.sum(df_I['gi'] * np.exp(-df_I['Ei'] / (kb_eV * T)))
    U_II = np.sum(df_II['gi'] * np.exp(-df_II['Ei'] / (kb_eV * T)))

    return U_I, U_II

# -----------------------------
# Load experiment metadata
# -----------------------------
def load_metadata():
    sample_list = pd.read_csv(SAMPLE_LIST_PATH, sep='	', decimal='.')
    eion = pd.read_csv(EION_PATH, sep='	', decimal='.')
    quant_par = pd.read_csv(QUANT_PAR_PATH, sep='	', decimal='.')
    line_data = pd.read_csv(LINE_DATA_PATH, sep='	', decimal='.')
    line_data.dropna(inplace=True)
    return sample_list, eion, quant_par, line_data

# -----------------------------
# Main execution block
# -----------------------------
def main():
    print("This is a template script. Please update file paths and implement analysis logic.")

if __name__ == '__main__':
    main()
