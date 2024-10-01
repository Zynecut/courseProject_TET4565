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
    model = modelSetup(data)
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


scenario_probabilities = {'low': 0.3, 'med': 0.4, 'high': 0.3}
def ObjFunction(model):
    day_ahead = sum(model.P_1[g, 1, 'med'] * model.MC[g] for g in model.Gen) + model.P_res['hydro']*model.MC_res['hydro']
    # balance_market = sum(scenario_probabilities[s]*sum(model.P_1[g, 2, s]*model.MC[g]for g in model.Gen + model.P_rat[s] * model.MC_rat) for s in model.Scenario)
    return day_ahead# + balance_market


def LoadBalanceDA(model):
    return sum(model.P_1[g, 1, s] for g in model.Gen for s in model.Scenario) == model.Load

def WindBalanceDA(model, s):
    return model.P_1['wind', 1, s] <= model.Wind[s]

def GenLimitDA(model, g, s):
    return model.P_min[g], model.P_1[g, 1, s], model.P_max[g]

def HydroReserveUseConstraint(model, s):
    return model.P_1['hydro', 1, s] <= model.P_max['hydro'] - model.P_res['hydro']

# def NuclearNonAnticipativity(model, s):
#     return model.P_1['nuclear', 2, s] == model.P_1['nuclear', 1, s]

# def WindSpilled(model, s):
#     return model.WS[2, s] == model.WindRT - model.P_1['wind', 2, s]

# def WindSpilledLimit(model, s):
#     return model.WS[2, s] <= model.WindRT

# def PowerBalanceRT(model, s):
#     return model.dH[s] + model.P_rat[s] + (model.WindRT - model.Wind[s] - model.WS[s]) == 0

# def RationingLimit(model, s):
#     return model.P_rat[s] <= model.Load



def modelSetup(data):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    model.Gen = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    model.Scenario = pyo.Set(initialize=['low', 'med', 'high'])  # ('low', 'med', 'high')
    # model.Stage = pyo.Set(initialize=[1, 2])  # ('1', '2')

    """
    Parameters
    """
    model.P_max = pyo.Param(model.Gen, initialize=data['Producers']['p_max'])
    model.P_min = pyo.Param(model.Gen, initialize=data['Producers']['p_min'])
    model.MC = pyo.Param(model.Gen, initialize=data['Producers']['marginal_cost'])
    model.P_res = pyo.Param(model.Gen, initialize=data['Producers']['p_res'])
    model.MC_res = pyo.Param(model.Gen, initialize=data['Producers']['reserve_cost'])

    model.Load = pyo.Param(initialize=data['Consumers']['consumption'].values())
    model.MC_rat = pyo.Param(initialize=data['Consumers']['rationing_cost'].values())

    model.Wind = pyo.Param(model.Scenario, initialize=data['Time_wind'])
    model.WindRT = pyo.Param(initialize=40)

    """
    Stage 1 Variables
    """
    model.P_1 = pyo.Var(model.Gen, model.Stage, model.Scenario, within=pyo.NonNegativeReals)    # Power production
    # Sjekk iPad, følg matten

    """
    Stage 2 Variables
    """
    # model.P_rat = pyo.Var(model.Scenario, within=pyo.NonNegativeReals)   # Rationing
    # model.dH = pyo.Var(model.Scenario, within=pyo.NonNegativeReals)     # Change in hydro production
    # model.WS = pyo.Var(model.Scenario, within=pyo.NonNegativeReals)                # Wind spillage
    # model.P_2 = pyo.Var(model.Gen, model.Scenario, within=pyo.NonNegativeReals)                 # Change in power production

    """
    Objective Function
    """
    model.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)


    """
    Constraints
    """
    model.LoadBalanceDA = pyo.Constraint(rule=LoadBalanceDA)
    model.WindBalanceDA = pyo.Constraint(model.Scenario, rule=WindBalanceDA)
    model.GenLimitDA = pyo.Constraint(model.Gen, model.Scenario, rule=GenLimitDA)
    model.HydroReserveUseConstraint = pyo.Constraint(model.Scenario, rule=HydroReserveUseConstraint)
    # model.NuclearNonAnticipativity = pyo.Constraint(model.Scenario, rule=NuclearNonAnticipativity)
    # model.WindSpilled = pyo.Constraint(model.Scenario, rule=WindSpilled)
    # model.WindSpilledLimit = pyo.Constraint(model.Scenario, rule=WindSpilledLimit)
    # model.PowerBalanceRT = pyo.Constraint(model.Scenario, rule=PowerBalanceRT)
    # model.RationingLimit = pyo.Constraint(model.Scenario, rule=RationingLimit)

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