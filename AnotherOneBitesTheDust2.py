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


def ObjFunction(model):
    production_cost = sum(model.P[g, s] * model.MC[g] for g in model.Gen for s in model.Scenario)
    rationing_cost = sum(model.MC_rat[l] * model.Rationing[l] for l in model.Load)
    return production_cost + rationing_cost


def LoadBalanceDA(model):
    for s in model.Scenario:
        return sum(model.P[g, s] for g in model.Gen) + sum(model.Rationing[l] for l in model.Load) == model.demand['Load 1']

# Beregner forventet vindproduksjon
def expected_wind_production(model):
    return sum(model.P_wind[s] * model.probabilities[s] for s in model.Scenario)  # Beregner forventet vindproduksjon basert på sannsynlighetene for de ulike scenarioene.


def wind_limitation(model, s):
    return model.P['wind', s] <= model.P_wind[s]  # Begrenser vindproduksjon til kalkulert forventning.




def modelSetup_1(data):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    model.Gen =         pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    model.Scenario =    pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')
    model.Load =        pyo.Set(initialize=list(data['Consumers']['nodeID'].keys()))  # ('Load 1')

    """
    Parameters
    """
    model.P_max =       pyo.Param(model.Gen, initialize=data['Producers']['p_max'])
    model.P_min =       pyo.Param(model.Gen, initialize=data['Producers']['p_min'])
    model.MC =          pyo.Param(model.Gen, initialize=data['Producers']['marginal_cost'])
    model.P_res =       pyo.Param(model.Gen, initialize=data['Producers']['p_res'])
    model.MC_res =      pyo.Param(model.Gen, initialize=data['Producers']['reserve_cost'])

    model.demand =      pyo.Param(model.Load, initialize=data['Consumers']['consumption'])
    model.MC_rat =      pyo.Param(model.Load, initialize=data['Consumers']['rationing_cost'])

    model.P_wind = pyo.Param(model.Scenario, initialize=data['Time_wind'])
    model.probabilities = pyo.Param(model.Scenario, initialize={'low': 0.3, 'med': 0.4, 'high': 0.3})

    # model.expected_wind_production = pyo.Expression(rule=expected_wind_production)

    """
    Stage 1 Variables
    """
    def p_bounds(model, g, s):
        return model.P_min[g], model.P_max[g]
    model.P = pyo.Var(model.Gen, model.Scenario, bounds= p_bounds, within=pyo.NonNegativeReals)    # Power production

    def rationing_bounds(model,l):
        return 0, model.demand[l]
    model.Rationing = pyo.Var(model.Load, bounds=rationing_bounds, within=pyo.NonNegativeReals)    # Rationing

    """
    Constraints
    """
    model.LoadBalanceDA = pyo.Constraint(rule=LoadBalanceDA)
    model.WindLimitation = pyo.Constraint(model.Scenario, rule=wind_limitation)

    """
    Objective Function
    """
    model.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    return model






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