from pyomo.environ import *
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def inputData(file):
    """
    Reads data from the Excel file and returns a dictionary containing data from relevant sheets.

    Parameters:
    file (str): The path to the Excel file.

    Returns:
    dict: A dictionary containing data from each sheet as lists of dictionaries.
    """
    data = {}
    excel_sheets = ['Producers', 'Consumers', 'Time_wind']
    for sheet in excel_sheets:
        df = pd.read_excel(file, sheet_name=sheet)
        df.index += 1
        data[sheet] = df.to_dict()
    return data


file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
data = inputData(file_name)


# Mapping mellom produsent-IDer og produsentnavn.
producer_map = {i: data['Producers']['type'][i] for i in data['Producers']['type'].keys()}

# Parametere fra data, mappe produsent-ID til navn
P_max = {producer_map[i]: data['Producers']['p_max'][i] for i in data['Producers']['producer'].keys()}              # Maksimal kapasitet
MC_gen = {producer_map[i]: data['Producers']['marginal_cost'][i] for i in data['Producers']['producer'].keys()}     # Marginalkostnader
MC_res = {producer_map[i]: data['Producers']['reserve_cost'][i] for i in data['Producers']['producer'].keys()}      # Reservekostnader
C_rationing = data['Consumers']['rationing cost'][1]    # Rationing cost
demand = data['Consumers']['consumption'][1]            # Total demand

PR_lim = 20

# Wind scenarios
wind_scenarios = {
    1: data['Time_wind']['wind_low'][1],
    2: data['Time_wind']['wind_med'][1],
    3: data['Time_wind']['wind_high'][1]
}

# Define probabilities for the three scenarios (low, med, high)
prob_s = {1: 1/3, 2: 1/3, 3: 1/3}


model = ConcreteModel()

# Sets
model.G = Set(initialize=producer_map.values())  # Generatorer
model.S = Set(initialize=[1, 2, 3])              # Vindscenarioer