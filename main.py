# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from datetime import datetime
from collections import defaultdict

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


def ObjFunction(m):
    return sum(m.mc[g]*m.gen[g] for g in m.Producers)

def GenerationLimit(m, g):
    return m.pmin[g], m.gen[g], m.pmax[g]

def WindGenerationMed(m, g):
    if m.source == 'wind':
        # return m.gen[g] <= m.pmax[g]*m.wind_med[g]*m.stochastic_period[g]
        return m.gen[g] <= m.pmax[g]*m.wind_med[g]

# Load constraints (matching consumer demand)
def LoadLimit(m, l):
    return sum(m.gen[g] for g in m.Producers) >= m.Demand[l]


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
    # m.Nodes = pyo.Set(initialize=xxx)  # Nodes

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
    # Wind data is given as a ratio of the maximum power output
    m.wind_low           = pyo.Param(m.Producers, initialize=data['Time_wind']['wind_low'])          # Wind low
    m.wind_med           = pyo.Param(m.Producers, initialize=data['Time_wind']['wind_med'])          # Wind medium
    m.wind_high          = pyo.Param(m.Producers, initialize=data['Time_wind']['wind_high'])         # Wind high
    m.stochastic_period  = pyo.Param(m.Producers, initialize=data['Time_wind']['stochastic_period']) # Stochastic period

    """
    Variables
    """
    m.gen = pyo.Var(m.Producers, within=pyo.NonNegativeReals)  # Power output

    """
    Constraints
    """
    m.GenLimit_Const = pyo.Constraint(m.Producers, rule=GenerationLimit)    # Power output constraint
    m.windGen_Const = pyo.Constraint(m.Producers, rule=WindGeneration)      # Wind generation constraint
    m.Load_Const = pyo.Constraint(m.Consumers, rule=LoadLimit)              # Load constraint
    """
    Objective Function
    """
    # Define objective function
    m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    return(m)

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


