# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

def main():
    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    benders(data)

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

    df_prod = pd.DataFrame(data['Producers'])
    df_prod = df_prod.set_index('type')
    data['Producers'] = df_prod.to_dict()

    df_load = pd.DataFrame(data['Consumers'])
    df_load = df_load.set_index('load')
    data['Consumers'] = df_load.to_dict()

    df = pd.DataFrame(data['Time_wind'])
    df = df.set_index('Stage')
    df = df.reset_index(drop=True)
    data['Time_wind'] = df.iloc[0].to_dict()

    return data

const = {'MC_res': 25}

def Obj_1st(m):
    production_cost_DA = m.hydro_res_DA * const['MC_res'] + m.hydro_DA * m.MC['hydro'] + m.nuclear_DA * m.MC['nuclear']
    return production_cost_DA + m.alpha

"""Set Variable Bounds"""
def limit_nuclear_DA(m):
    return m.P_min["nuclear"], m.P_max["nuclear"]  # Begrens produksjonen til maksimal produksjon for kjernekraft.

def limit_hydro_DA(m):
    return m.P_min["hydro"], m.P_max["hydro"]

def load_balance_DA(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand["Load 1"]

def hydro_res_min(m):
    return m.hydro_res_DA >= 0

def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] + m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])

def load_balance_RT(m, s):
    """
    Load balance constraint for real-time market, considering scenario s.
    """
    return m.nuclear_DA + m.hydro_DA + m.P_wind[s] == m.demand["Load 1"]

def Model_1st(data, Cuts):
    m = pyo.ConcreteModel()

    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')

    """Parameters"""
    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.wind_DA = pyo.Param(initialize=50)


    """Variables"""
    m.nuclear_DA = pyo.Var(bounds=limit_nuclear_DA, within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(bounds=limit_hydro_DA, within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.wind_prod_DA = pyo.Var(within=pyo.NonNegativeReals)  # Vindproduksjon i day-ahead

    """Cuts"""
    m.Cut = pyo.Set(initialize=Cuts["Set"])  # Set for cuts
    m.Phi = pyo.Param(m.Cut, initialize=Cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda = pyo.Param(m.Cut, initialize=Cuts["lambda"])  # Parameter for lambda (dual value of reservoir)
    m.X_hat = pyo.Param(m.Cut, initialize=Cuts["x_hat"])  # Parameter for reservoir level

    m.alpha = pyo.Var(bounds=(-100000, 100000))  # Variable for alpha

    """Constraints"""
    m.LoadBalance_DA = pyo.Constraint(rule=load_balance_DA)
    m.HydroResMin = pyo.Constraint(rule=hydro_res_min)
    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)

    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)

    return m

def Model_2nd(data, X_hat, wind):
    m = pyo.ConcreteModel()

    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')
    m.S = pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')

    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.cost_rat = pyo.Param(m.L, initialize=data['Consumers']['rationing cost'])
    m.P_wind = pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob = pyo.Param(m.S, initialize={'low': 1 / 3, 'med': 1 / 3, 'high': 1 / 3})
    m.X_hat = pyo.Param(initialize=X_hat)

    m.nuclear_DA = pyo.Var(bounds=limit_nuclear_DA, within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(bounds=limit_hydro_DA, within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.wind_prod_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.LoadBalance_RT = pyo.Constraint(m.S, rule=load_balance_RT)

    """Definer dual suffix"""
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)


    return m

# Solve function
def SolveModel(model):
    opt = SolverFactory("gurobi")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # results = opt.solve(model, load_solutions=True)
    results = opt.solve(model, options={"ResultFile": "infeasibility.ilp"})
    return results, model

def benders(data):
    # Initielle kutt og parametere
    Cuts = {"Set": [], "Phi": {}, "lambda": {}, "x_hat": {}}

    # Iterasjonsloop
    for iteration in range(1, 100):
        print(f"Iteration: {iteration}")

        # Første-stegs modell (master problem)
        model_1st = Model_1st(data, Cuts)
        SolveModel(model_1st)

        # Lagre X_hat (hydro_res_DA verdi) fra første stegs løsning
        X_hat = pyo.value(model_1st.hydro_res_DA)

        # Andre-stegs modell (subproblem)
        wind_scenario = {}  # Sett vindscenario her
        model_2nd = Model_2nd(data, X_hat, wind_scenario)
        results, model_2nd = SolveModel(model_2nd)

        # Hent dualvariabel for LoadBalance_RT constraint
        dual_lambda = model_2nd.dual[model_2nd.LoadBalance_RT]

        # Generer kutt basert på dualvariabler
        # Oppdater kutt-settet med verdier
        Cuts["Set"].append(iteration)
        Cuts["Phi"][iteration] = ...  # Sett korrekt Phi basert på subproblem-resultat
        Cuts["lambda"][iteration] = dual_lambda
        Cuts["x_hat"][iteration] = X_hat

if __name__ == '__main__':
    main()
