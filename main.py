# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *

from textdistance import mlipns

# Structure

"""
generator data: capacity, marg_cost, location, slackbus, CO2_emission

load data: demand, marg_WTP, location

transmission data, capacity, susceptance


Datasett Producers comment
Fuelcost neglected for wind, solar and hydro
Inflow_factor neglected (hydro)
Storage price neglected. (hydro)
Initial storage neglected. (hydro)
Marginal_cost is just a wild guess


Datasett Consumers comment
Add flexible consumers (WTP)?


Datasett Time comment
Load is given as a ratio of demand
Wind is given as a ratio of p_max
Solar is given as a ratio of p_max


Datasett Node comment
Only for geographical plotting 

"""

def main():

    file_name = 'Datasett_NO1_Cleaned_r4.xlsx'
    data = inputData(file_name)
    m = modelSetup(data)
    results, m = SolveModel(m)
    DisplayModelResults(m)
    return()



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


def average_wind(wind_data, day_indices):
    """ Compute the average wind power over a set of time indices """
    total_wind = sum(wind_data[time] for time in day_indices)
    return total_wind / len(day_indices)



def ObjFunction(m):
    return sum(m.mc[g]*m.gen[g] for g in m.Producers)

# Define probabilities for the three scenarios (low, med, high)
scenario_probabilities = {'low': 1/3, 'med': 1/3, 'high': 1/3}
def StochasticObjFunction(m):
    return sum(scenario_probabilities[s] * sum(m.mc[g] * m.gen[g] for g in m.Producers) for s in m.Scenarios)



# def GenerationLimit(m, g):
#    return m.pmin[g], m.gen[g], m.pmax[g]

# Load constraints (matching consumer demand)
def LoadLimit(m, l):
    return sum(m.gen[g] for g in m.Producers) >= m.Demand[l]


def HydroGenerationLimit(m, g):
    if m.source[g] == 'hydro':
        return m.pmin[g], m.gen[g], m.pmax[g]
    else:
        return Constraint.Skip  # Skip for non-hydro generators


# Deterministic wind generation for the first two days
def WindGenerationDeterministic(m, g):
    if m.source[g] == 'wind':  # For the first two days
        return m.pmin[g], m.gen[g], m.pmax[g]*m.wind_avg
    else:
        return Constraint.Skip  # Skip for the third day (handled by scenarios)


"""
Det som ikke funker her er at gen er jo mye større enn pmax, siden den allerede har dratt inn hydro??
"""


# Stochastic wind generation for the third day
def WindGenerationStochastic(m, g, s):
    if m.source[g] == 'wind':  # For the third day (stochastic period)
        if s == 'low':
            return m.pmin[g], m.gen[g], m.pmax[g]*m.wind_avg_low
        elif s == 'high':
            return m.pmin[g], m.gen[g], m.pmax[g]*m.wind_avg_high
        else:
            return m.pmin[g], m.gen[g], m.pmax[g]*m.wind_avg_med  # Medium scenario for the third day
    else:
        return Constraint.Skip


def modelSetup(data):# , b_matrix):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    m.Producers = pyo.Set(initialize=list(data['Producers']['nodeID']))  # Generators
    m.Consumers = pyo.Set(initialize=list(data['Consumers']['load']))  # Load units
    # m.Time = pyo.Set(initialize=[1, 2, 3])  # Time periods
    m.Scenarios = pyo.Set(initialize=['low', 'med', 'high'])  # Scenarios

    """
    Parameters
    """
    # Generator data
    m.pmax          = pyo.Param(m.Producers, initialize=data['Producers']['pmax'])              # Max power output
    m.pmin          = pyo.Param(m.Producers, initialize=data['Producers']['pmin'])              # Min power output
    m.mc            = pyo.Param(m.Producers, initialize=data['Producers']['marginal_cost'])     # Marginal cost
    m.storage_cap   = pyo.Param(m.Producers, initialize=data['Producers']['storage_cap'])       # Storage capacity
    m.source        = pyo.Param(m.Producers, initialize=data['Producers']['gen_source'])        # Type of generator

    # Load data
    m.Demand        = pyo.Param(m.Consumers, initialize=data['Consumers']['consumption'])       # Demand
    m.Rationing     = pyo.Param(m.Consumers, initialize=data['Consumers']['Rationing cost'])    # Rationing

    # Wind data
    # m.wind_low           = pyo.Param(m.Time, initialize=data['Time_wind']['wind_low'])          # Wind low
    # m.wind_med           = pyo.Param(m.Time, initialize=data['Time_wind']['wind_med'])          # Wind medium
    # m.wind_high          = pyo.Param(m.Time, initialize=data['Time_wind']['wind_high'])         # Wind high

    day_12_avg = average_wind(data['Time_wind']['wind_med'], [1, 2, 3, 4, 5, 6, 7, 8])
    m.wind_avg = pyo.Param(initialize=day_12_avg)

    # day_3_avg_low = average_wind(data['Time_wind']['wind_low'], [9, 10, 11, 12, 13])
    # day_3_avg_med = average_wind(data['Time_wind']['wind_med'], [9, 10, 11, 12, 13])
    # day_3_avg_high = average_wind(data['Time_wind']['wind_high'], [9, 10, 11, 12, 13])
    # m.wind_avg_low = pyo.Param(m.Producers, initialize=day_3_avg_low)
    # m.wind_avg_med = pyo.Param(m.Producers, initialize=day_3_avg_med)
    # m.wind_avg_high = pyo.Param(m.Producers, initialize=day_3_avg_high)

    """
    Variables
    """
    m.gen = pyo.Var(m.Producers, within=pyo.NonNegativeReals)  # Power output

    """
    Constraints
    """
    # m.GenLimit_Const = pyo.Constraint(m.Producers, rule=GenerationLimit)    # Power output constraint

    # Hydro generation (no stochastic behavior)
    m.HydroGen_Const = pyo.Constraint(m.Producers, rule=HydroGenerationLimit)

    # Deterministic wind generation for the first two days
    m.WindGen_Det = pyo.Constraint(m.Producers, rule=WindGenerationDeterministic)

    # Stochastic wind generation for the third day
    # m.WindGen_Sto = pyo.Constraint(m.Producers, m.Scenarios, rule=WindGenerationStochastic)

    """
    Må slå sammen Det og Sto??
    """

    # Load constraint
    m.Load_Const = pyo.Constraint(m.Consumers, rule=LoadLimit)              # Load constraint

    """
    Objective Function
    """
    # Define objective function
    m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)
    # m.obj = pyo.Objective(rule=StochasticObjFunction, sense=pyo.minimize)

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


