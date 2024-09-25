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
    return

# Define probabilities for the three scenarios (low, med, high)
scenario_probabilities = {'low': 1/3, 'med': 1/3, 'high': 1/3}



def modelSetup(data):# , b_matrix):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    m.P = pyo.Set(initialize=list(data['Producers']['type'].keys()))
    m.C = pyo.Set(initialize=list(data['Consumers']['load'].keys()))
    m.S = pyo.Set(initialize=['low', 'med', 'high'])
    m.K = pyo.Set(initialize=[1, 2])


    """
    Parameters
    """


    """
    Decision Variables
    """


    """
    Stochastic Variables
    """




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


