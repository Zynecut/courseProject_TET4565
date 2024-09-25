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
producer_map = {i: data['Producers']['producer'][i] for i in data['Producers']['producer'].keys()}

# Parametere fra data, mappe produsent-ID til navn
P_max = {producer_map[i]: data['Producers']['p_max'][i] for i in data['Producers']['producer'].keys()}  # Maksimal kapasitet
c = {producer_map[i]: data['Producers']['marginal_cost'][i] for i in data['Producers']['producer'].keys()}  # Marginalkostnader
c_R = {producer_map[i]: data['Producers']['reserve_cost'][i] for i in data['Producers']['producer'].keys()}  # Reservekostnader

PR_lim = 20


# Wind scenarios
wind_scenarios = {
    1: data['Time_wind']['wind_low'][1],
    2: data['Time_wind']['wind_med'][1],
    3: data['Time_wind']['wind_high'][1]
}

# Define probabilities for the three scenarios (low, med, high)
prob_s = {1: 1/3, 2: 1/3, 3: 1/3}

load = data['Consumers']['consumption'][1]  # Total demand
c_rat = data['Consumers']['rationing cost'][1]  # Rationing cost

model = ConcreteModel()

# Sets
model.G = Set(initialize=producer_map.values())  # Generatorer
model.S = Set(initialize=[1, 2, 3])              # Vindscenarioer

# Variables
# Firs stage (day-ahead) decisions
model.p = Var(model.G, domain=NonNegativeReals)             # Produksjon fra nuclear og hydro
model.R_hydro = Var(domain=NonNegativeReals, bounds =(0,PR_lim))     # Reservekapasitet for hydro. Begrenser denne til "PR_lim".

# Second stage (real-time) (scenario dependent) decisions
model.r = Var(model.G, model.S, domain=NonNegativeReals)    # Regulering for hvert scenario og generator
model.P_rat = Var(model.S, domain=NonNegativeReals)         # Rasjonering i hvert scenario



# Minimize first stage cost + expected second stage cost
def objective_rule(model):
    first_stage_cost = sum(c[g] * model.p[g] for g in model.G) + sum(c_R.get(g, 0) * model.R_hydro for g in model.G if g == 'Producer 2')  # Hydro producer
    second_stage_cost = sum(prob_s[s] * (sum(c[g] * model.r[g, s] for g in model.G) + c_rat * model.P_rat[s]) for s in model.S)
    return first_stage_cost + second_stage_cost

model.objective = Objective(rule=objective_rule, sense=minimize)

# Constraints

def power_balance_rule(model, s):
    return sum(model.p[g] + model.r[g, s] for g in model.G) + wind_scenarios[s] + model.P_rat[s] == load
model.power_balance = Constraint(model.S, rule=power_balance_rule)


# 2. Production limit for each generator
def production_limit_rule(model, g):
    return model.p[g] <= P_max[g]
model.production_limit = Constraint(model.G, rule=production_limit_rule)


def hydro_limit_rule(model):
    # Hydro kan produsere opptil 40 enheter i day-ahead, men totale kapasiteten er fortsatt 60 enheter
    return model.p['Producer 2'] + model.R_hydro <= P_max['Producer 2']  # Hydro har totalt 60 enheter tilgjengelig (20 reservert)
# model.hydro_limit = Constraint(rule=hydro_limit_rule)


# 4. Regulation limit for each generator
def regulation_limit_rule(model, g, s):
    if g == 'Producer 1':                               # Nuclear
        return model.r[g, s] == 0                       # Nuclear har ingen reguleringskapasitet
    elif g == 'Producer 2':                             # Hydro
        return model.r[g, s] <= model.R_hydro           # Hydro kan regulere innenfor reserver
    elif g == 'Producer 3':                             # Wind
        return model.r[g, s] == 0                       # Wind har ingen reguleringskapasitet
# model.regulation_limit = Constraint(model.G, model.S, rule=regulation_limit_rule)

# 5. Rationing limit can not exceed total demand
def rationing_limit_rule(model, s):
    return model.P_rat[s] <= load
# model.rationing_limit = Constraint(model.S, rule=rationing_limit_rule)

solver = SolverFactory('glpk')  # Du kan også bruke 'cbc' eller en annen solver
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
result = solver.solve(model, tee=True)

# Vise resultater
model.p.display()
model.R_hydro.display()
model.r.display()
model.P_rat.display()
model.dual.display()