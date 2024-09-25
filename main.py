# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# Structure

"""
Stochasticity in linear programming

Hard constraints -> Robust optimization
Soft constraints -> Chance constraints


"""

def main():

    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    m = modelSetup(data)
    results, m = SolveModel(m)
    DisplayModelResults(m)
    # return()



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

    df_nuke = pd.DataFrame(data['Producers'])
    df_nuke = df_nuke.set_index('type')
    data['Producers'] = df_nuke.to_dict()

    df = pd.DataFrame(data['Time_wind'])
    df = df.set_index('Stage')
    df = df.reset_index(drop=True)
    data['Time_wind'] = df.iloc[0].to_dict()

    return data

# Define probabilities for the three scenarios (low, med, high)

scenario_probabilities = {'low': 0.2, 'med': 0.5, 'high': 0.3}
def ObjFunction(m):
    return sum(m.p_1[g]*m.mc[g] for g in m.G)  + \
           sum(scenario_probabilities[s] * sum(m.p_2[g]*m.mc[g] for g in m.G) for s in m.S)


def PowerBalanceStage1(m, l):
    return sum(m.p_1[g] for g in m.G) + m.w_s['med'] - m.demand[l] == 0


def GenerationLimit(m, g):
    return m.p_1[g] <= m.p_max[g]


def WindLimitation(m, s):
    return m.w_s[s] <= m.wind[s]



def NonAnticipativity(m, g):
    return m.p_1[g] - m.p_2[g] == 0

def PowerBalanceStage2(m, l):
    return sum(m.p_2[g] for g in m.G) + m.wind_actual - m.demand[l] == 0


def modelSetup(data):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))
    m.L = pyo.Set(initialize=list(data['Consumers']['load'].keys()))
    m.S = pyo.Set(initialize=['low', 'med', 'high'])
    m.K = pyo.Set(initialize=[1, 2])


    """
    Parameters
    """
    # Producer data
    m.p_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])                 # Maximum power
    m.p_res = pyo.Param(m.G, initialize=data['Producers']['p_res'])                 # Reserve power
    m.res_cost = pyo.Param(m.G, initialize=data['Producers']['reserve_cost'])       # Reserve cost
    m.mc = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])            # Marginal cost

    # Consumer data
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])              # Demand
    m.rationing_cost = pyo.Param(m.L, initialize=data['Consumers']['rationing_cost'])   # Rationing cost

    # Wind data
    m.wind = pyo.Param(m.S, initialize=data['Time_wind'])   # Wind forecast scenarios
    m.wind_actual = pyo.Param(initialize=30)                        # Actual wind generation
    m.wind.pprint()
    """
    Decision Variables
    """
    m.p_1 = pyo.Var(m.G, within=pyo.NonNegativeReals)       # Power Day Ahead
    # m.r_1 = pyo.Var(m.G, within=pyo.NonNegativeReals)       # Reserve Day Ahead
    m.w_s = pyo.Var(m.S, within=pyo.NonNegativeReals)       # Wind Forecast Day Ahead with scenarios
    """
    Stochastic Variables
    """
    m.p_2 = pyo.Var(m.G, within=pyo.NonNegativeReals)  # Power Real Time


    """
    Objective Function
    """
    m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    """
    Constraints
    """
    # Stage 1
    m.GenLim = pyo.Constraint(m.G, rule=GenerationLimit)
    m.WindLim = pyo.Constraint(m.S, rule=WindLimitation)
    m.PowerBalanceStage1 = pyo.Constraint(m.L, rule=PowerBalanceStage1)

    # Stage 2
    m.NonAnticipativity = pyo.Constraint(m.G, rule=NonAnticipativity)
    m.PowerBalanceStage2 = pyo.Constraint(m.L, rule=PowerBalanceStage2)

    return m

# Solve function
def SolveModel(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m


# Display results
def DisplayModelResults(m):
    # return m.pprint()
    return print(m.display(), m.dual.display())




if __name__ == '__main__':
    main()


