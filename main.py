# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Structure

"""
generator data: capacity, marg_cost, location, slackbus, CO2_emission

load data: demand, marg_WTP, location

transmission data, capacity, susceptance
"""

def main():
    # read data from excel files
    file_name = 'data.xlsx'
    data = read_data(file_name)
    B_matrix = make_Bmatrix(data)
    DCOPF_Model(data, B_matrix)
    return data

def read_data(file):
    # read data from file
    data = {}
    df = pd.read_excel
    return data
    
    
def make_Bmatrix(data):
    length = len(data[task]['Line']['Susceptance [p.u]'])               # Get the length of the B matrix
    B_matrix = np.zeros((length, length))                               # Create empty matrix
    return B_matrix
    
def fix_Data(data):
    # fix the data to be in the right format
    return data
    
def DCOPF_Model(data, B_matrix):
    """
    Setup the optimization model, run it and store the data in a .xlsx file
    """
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """


    """
    Parameters
    """


    """
    Variables
    """


    """
    Objective Function
    """


    return()


if __name__ == '__main__':
    main()


