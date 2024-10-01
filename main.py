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
    DisplayResultsDataFrame(m)
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






def GenerationLimit(m, g, k, s):
    return m.p_min[g], m.p_1[g, k, s], m.p_max[g]

def WindLimitation(m, k, s):
    return m.p_1['wind', k, s] <= m.wind[s]

def LockNuclearProduction(m, s):
    return m.p_1['nuclear', 1, s] == m.p_1['nuclear', 2, s]

def AdjustHydroRealTimeLB(m, k, s):
    if k == 1:
        return pyo.Constraint.Skip
    else:
        return m.p_1['hydro', 1, s] - m.p_res['hydro'] <= m.p_1['hydro', k, s]

def AdjustHydroRealTimeUB(m, k, s):
    if k == 1:
        return pyo.Constraint.Skip
    else:
        return m.p_1['hydro', 1, s] + m.p_res['hydro'] >= m.p_1['hydro', k, s]


def LoadBalanceStage1(m, l, k, s):
    return sum(m.p_1[g, k, s] for g in m.G) + m.r_1[l, k, s] - m.demand[l] == 0








# def NonAnticipativity(m, g):
#     return m.p_1[g] - m.p_2[g] == 0

def PowerBalanceStage2(m, l):
    return sum(m.p_2[g] for g in m.G) + m.wind_actual - m.demand[l] == 0


# Define probabilities for the three scenarios (low, med, high)
scenario_probabilities = {'low': 0.2, 'med': 0.5, 'high': 0.3}
def ObjFunction(m):
    production_cost = sum(m.p_1[g, k, s] * m.mc[g] for g in m.G for k in m.K for s in m.S)
    reserved_cost = sum(m.p_res['hydro'] * m.res_cost['hydro'] for k in m.K)
    rationing_cost = sum(m.r_1[l, k, s] * m.rationing_cost[l] for l in m.L for k in m.K for s in m.S)

    return production_cost + reserved_cost + rationing_cost




def modelSetup(data):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))       # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['nodeID'].keys()))        # ('Load 1')
    m.S = pyo.Set(initialize=['low', 'med', 'high'])                        # ('low', 'med', 'high')
    m.K = pyo.Set(initialize=[1, 2])                                        # ('1', '2')


    """
    Parameters
    """
    # Producer data
    m.p_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])                 # Maximum power {nuclear: 200, hydro: 60, wind: 80}
    m.p_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])                 # Minimum power {nuclear: 0, hydro: 0, wind: 0}
    m.p_res = pyo.Param(m.G, initialize=data['Producers']['p_res'])                 # Reserve power {nuclear: 0, hydro: 20, wind: 0}
    m.res_cost = pyo.Param(m.G, initialize=data['Producers']['reserve_cost'])       # Reserve cost  {nuclear: 0, hydro: 30, wind: 0}
    m.mc = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])            # Marginal cost {nuclear: 15, hydro: 30, wind: 0}

    # Consumer data
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])              # Demand    {'Load 1': 250}
    m.rationing_cost = pyo.Param(m.L, initialize=data['Consumers']['rationing_cost'])   # Rationing cost {'Load 1': 100}

    # Wind data
    m.wind = pyo.Param(m.S, initialize=data['Time_wind'])   # Wind forecast scenarios   {'low': 34.5, 'med': 45.6, 'high': 55.9}
    m.wind_actual = pyo.Param(initialize=30)                        # Actual wind generation
    # m.wind.pprint()

    """
    Decision Variables
    """
    m.p_1 = pyo.Var(m.G, m.K, m.S, within=pyo.NonNegativeReals)       # Power Day Ahead
    m.r_1 = pyo.Var(m.L, m.K, m.S, within=pyo.NonNegativeReals)       # Reserve Day Ahead
    m.H_DA = pyo.Var(m.K, within=pyo.NonNegativeReals)                # Hydro storage Day Ahead
    m.N_DA = pyo.Var(m.K, within=pyo.NonNegativeReals)                # Nuclear storage Day Ahead

    """
    Stochastic Variables
    """
    # m.p_2 = pyo.Var(m.G, within=pyo.NonNegativeReals)  # Power Real Time


    """
    Objective Function
    """
    m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    """
    Constraints
    """
    # Stage 1
    m.GenLim = pyo.Constraint(m.G, m.K, m.S, rule=GenerationLimit)
    m.WindLim = pyo.Constraint(m.K, m.S, rule=WindLimitation)
    m.LockNuclear = pyo.Constraint(m.S, rule=LockNuclearProduction)
    m.AdjustHydroRTLB = pyo.Constraint(m.K, m.S, rule=AdjustHydroRealTimeLB)
    m.AdjustHydroRTUB = pyo.Constraint(m.K, m.S, rule=AdjustHydroRealTimeUB)

    m.PowerBalanceStage1 = pyo.Constraint(m.L, m.K, m.S, rule=LoadBalanceStage1)

    # Stage 2
    # .NonAnticipativity = pyo.Constraint(m.G, rule=NonAnticipativity)
    # m.PowerBalanceStage2 = pyo.Constraint(m.L, rule=PowerBalanceStage2)

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

def DisplayResultsDataFrame(m):
    results = []
    for k in m.K:
        for s in m.S:
            nuclear_prod = m.p_1['nuclear', k, s].value
            hydro_prod = m.p_1['hydro', k, s].value
            wind_prod = m.p_1['wind', k, s].value
            rationing_value = m.r_1['Load 1', k, s].value
            reserved_hydro = m.p_res['hydro']
            wind_spilled = m.wind[s] - wind_prod  # Beregn vind som ikke blir produsert
            results.append({
                'Stage': k,
                'Scenario': s,
                'Nuclear Production (MW)': nuclear_prod,
                'Hydro Production (MW)': hydro_prod,
                'Wind Production (MW)': wind_prod,
                'Wind Spilled (MW)': wind_spilled,
                'Rationing (MW)': rationing_value,
                'Hydro Reserved for Next Day (MW)': reserved_hydro
            })

    # Present results
    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    print(df_results)
    return()


if __name__ == '__main__':
    main()


