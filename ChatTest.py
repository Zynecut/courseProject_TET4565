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


# Objective function
def objective_rule(model):
    return (
            sum(model.c[k] * model.p[k] + model.cu[k] * model.Ru[k] + model.cd[k] * model.Rd[k] for k in model.G)
            + sum(model.pi[s] * (sum(model.c[k] * (model.ru[k, s] - model.rd[k, s]) for k in model.G)
                                 + model.c_rat * model.p_rat[s]) for s in model.S)
    )

# Constraint 1: Balance in Day-ahead market
def da_balance_rule(model):
    return sum(model.p[k] for k in model.G) == model.L  # l is the demand
# Constraint 2: Wind forecast upper bound
def WindUB(model, s):
    return model.p['wind'] <= model.w_max[s]
# Constraint 3 & 4: Generator production limits
def PUB(model, k):
    return model.p[k] + model.Ru[k] <= model.p_max[k]
def PLB(model, k):
    return model.p[k] - model.Rd[k] >= 0
# Constraint 5: Balance in Real-time market
def rt_balance_rule(model, s):
    return (sum(model.ru[k, s] - model.rd[k, s] for k in model.G)
            + model.p_rat[s] + model.w_RT - model.p['wind'] - model.w_SP[s] == 0)
# Constraints 6 & 7: Up-regulation and down-regulation limits
def up_regulation_limit(model, k, s):
    return model.ru[k, s] <= model.Ru[k]
def down_regulation_limit(model, k, s):
    return model.rd[k, s] <= model.Rd[k]
# Constraint 8: Wind spillage limit
def wind_spillage_limit(model, s):
    return model.w_SP[s] <= model.w_RT - model.p['wind']
# Constraint 9: Real-time market production limit
def rt_market_limit(model, s):
    return model.p_rat[s] <= model.L


def modelSetup(data):
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    # Sets
    model.G = pyo.Set(initialize=['nuclear', 'hydro', 'wind'])  # NG = Number of generators
    model.S = pyo.Set(initialize=['low', 'med', 'high'])  # NS = Number of scenarios

    # Parameters
    model.c = pyo.Param(model.G, initialize={'nuclear': 15, 'hydro': 30, 'wind': 0})  # Generator cost
    model.cu = pyo.Param(model.G, initialize={'nuclear': 0, 'hydro': 30, 'wind': 0})  # Up-regulation cost
    model.cd = pyo.Param(model.G, initialize={'nuclear': 15, 'hydro': 30, 'wind': 0})  # Down-regulation cost
    model.c_rat = pyo.Param(initialize=250)  # rationing cost
    model.pi = pyo.Param(model.S, initialize={'low': 0.3, 'med': 0.4, 'high': 0.3})  # Probability of each scenario
    model.p_max = pyo.Param(model.G, initialize={'nuclear': 200, 'hydro': 60, 'wind': 80})  # Max production
    model.w_max = pyo.Param(model.S, initialize={'low': 34.5, 'med': 45.6, 'high': 55.9})
    model.L = pyo.Param(initialize=250)
    model.w_RT = pyo.Param(initialize=50)
    model.w_DA = pyo.Param(initialize=45)  # Wind forecast

    # Variables
    model.p = pyo.Var(model.G, within=pyo.NonNegativeReals)  # Power production
    model.Ru = pyo.Var(model.G, within=pyo.NonNegativeReals)  # Up-regulation reserve
    model.Rd = pyo.Var(model.G, within=pyo.NonNegativeReals)  # Down-regulation reserve

    model.ru = pyo.Var(model.G, model.S, within=pyo.NonNegativeReals)  # Real-time up-regulation
    model.rd = pyo.Var(model.G, model.S, within=pyo.NonNegativeReals)  # Real-time down-regulation
    model.p_rat = pyo.Var(model.S, within=pyo.NonNegativeReals)  # Real-time market production
    model.w_SP = pyo.Var(model.S, within=pyo.NonNegativeReals)  # Wind spillage



    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    """
    Constraints
    """
    model.da_balance = pyo.Constraint(rule=da_balance_rule)
    # Constraint 2: Wind forecast upper bound
    model.wind_upper_bound = pyo.Constraint(model.S, rule=WindUB)
    # Constraint 3 & 4: Generator production limits
    model.production_upper_bound = pyo.Constraint(model.G, rule=PUB)
    model.production_lower_bound = pyo.Constraint(model.G, rule=PLB)
    # Constraint 5: Balance in Real-time market
    model.rt_balance = pyo.Constraint(model.S, rule=rt_balance_rule)
    # Constraints 6 & 7: Up-regulation and down-regulation limits
    model.up_regulation_limit = pyo.Constraint(model.G, model.S, rule=up_regulation_limit)
    model.down_regulation_limit = pyo.Constraint(model.G, model.S, rule=down_regulation_limit)
    # Constraint 8: Wind spillage limit
    model.wind_spillage_limit = pyo.Constraint(model.S, rule=wind_spillage_limit)
    # Constraint 9: Real-time market production limit
    model.rt_market_limit = pyo.Constraint(model.S, rule=rt_market_limit)


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