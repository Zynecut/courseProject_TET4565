# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pprint

# Structure

"""
Stochasticity in linear programming

Hard constraints -> Robust optimization
Soft constraints -> Chance constraints


"""

def main():

    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    pprint.pprint(data, width=1)
    model = modelSetup_1(data)
    results, model = SolveModel(model)
    DisplayModelResults(model)


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
    # df_prod = df_prod.transpose()
    df_prod = df_prod.set_index('type')
    data['Producers'] = df_prod.to_dict()

    df_load = pd.DataFrame(data['Consumers'])
    # df_load = df_load.transpose()
    df_load = df_load.set_index('load')
    data['Consumers'] = df_load.to_dict()

    df = pd.DataFrame(data['Time_wind'])
    df = df.set_index('Stage')
    df = df.reset_index(drop=True)
    data['Time_wind'] = df.iloc[0].to_dict()

    return data

const = {'p_res' : 20,'MC_res' : 30}

"""Set Variable Bounds"""
def limit_nuclear_DA(m):
    return m.P_min["nuclear"], m.P_max["nuclear"]  # Begrenser produksjonen til maksimal produksjon for kjernekraft.
def limit_hydro_DA(m):
    return m.P_min["hydro"], m.P_max["hydro"]
def limit_wind_DA(m):
    return m.P_min["wind"], m.P_max["wind"]
def limit_nuclear_RT(m, s):
    return m.P_min["nuclear"], m.P_max["nuclear"]
def limit_hydro_RT(m, s):
    return m.P_min["hydro"], m.P_max["hydro"]
def limit_wind_RT(m, s):
    return m.P_min["wind"], m.P_max["wind"]
def rationing_limits(m, l, s):
    return 0, m.demand[l]
def reserved_DA(m):
    return 0,const['p_res']

"""Constraints"""

def locked_nuclear_prod(m, s):
    return m.nuclear_DA == m.nuclear_RT[s]
def load_balance_DA(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA >= m.demand["Load 1"]
def load_balance_RT(m, s):
    return m.hydro_RT[s] + m.wind_prod_RT[s] + m.nuclear_RT[s] + m.rationing["Load 1", s] >= m.demand["Load 1"]
def hydro_upper_RT(m, s):
    return m.hydro_RT[s] <= m.hydro_DA + m.hydro_res_DA
def hydro_lower_RT(m, s):
    return m.hydro_RT[s] >= m.hydro_DA - m.hydro_res_DA
def hydro_res(m):
    return m.hydro_res_DA == const['p_res']


"""Objective Function"""
def ObjFunction(m):
    production_cost_DA = m.hydro_res_DA * const['MC_res']
    production_cost_RT = sum(m.prob[s] * (m.MC['nuclear'] * m.nuclear_RT[s] + m.MC['hydro'] * m.hydro_RT[s]) for s in m.S)
    rationing_cost = sum(m.prob[s] * m.cost_rat[l] * m.rationing[l, s] for l in m.L for s in m.S)
    return production_cost_DA + production_cost_RT + rationing_cost


def modelSetup_1(data):
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case
    """Sets"""
    m.G =           pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L =           pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')
    m.S =           pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')

    """Parameters"""
    m.P_max =       pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min =       pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC =          pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand =      pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.cost_rat =    pyo.Param(m.L, initialize=data['Consumers']['rationing cost'])
    m.P_wind =      pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob =        pyo.Param(m.S, initialize={'low': 1/3, 'med': 1/3, 'high': 1/3})
    m.wind_DA = sum(m.prob[s] * m.P_wind[s] for s in m.S)

    """Variables"""
    m.nuclear_DA =  pyo.Var(bounds=limit_nuclear_DA, within=pyo.NonNegativeReals)
    m.hydro_DA =    pyo.Var(bounds=limit_hydro_DA, within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(bounds=reserved_DA, within=pyo.NonNegativeReals)
    m.wind_prod_DA = pyo.Var(within=pyo.NonNegativeReals)  # Wind production in day-ahead
    m.nuclear_RT =  pyo.Var(m.S, bounds=limit_nuclear_RT, within=pyo.NonNegativeReals)
    m.hydro_RT =    pyo.Var(m.S, bounds=limit_hydro_RT, within=pyo.NonNegativeReals)
    m.wind_prod_RT = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.rationing =   pyo.Var(m.L, m.S, bounds=rationing_limits, within=pyo.NonNegativeReals)  # Setter øvre og nedre begrensning for rasjonering direkte i variabelen ved initialisering.


    """Constraints"""
    m.LockedNuclearProd =       pyo.Constraint(m.S, rule=locked_nuclear_prod)
    m.LoadBalance_DA =          pyo.Constraint(rule=load_balance_DA)
    m.LoadBalance_RT =          pyo.Constraint(m.S, rule=load_balance_RT)
    m.HydroUpper_RT = pyo.Constraint(m.S, rule=hydro_upper_RT)  # Hydro_RT <= Hydro_DA + reserved capacity
    m.HydroLower_RT = pyo.Constraint(m.S, rule=hydro_lower_RT)
    m.WindProdDAConstraint = pyo.Constraint(expr=m.wind_prod_DA == m.wind_DA)
    m.WindProdRTConstraint = pyo.Constraint(m.S, rule=lambda m, s: m.wind_prod_RT[s] == m.P_wind[s])
    m.Hydroreserved_DA = pyo.Constraint(rule=hydro_res)


    """Objective Function"""
    m.obj =         pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    return m






# Solve function
def SolveModel(model):
    opt = SolverFactory("gurobi")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions=True)
    return results, model


# Display results
def DisplayModelResults(model):
    # return m.pprint()
    return print(model.display(), model.dual.display())


if __name__ == '__main__':
    main()

