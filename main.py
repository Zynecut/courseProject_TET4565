# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from datetime import datetime
from collections import defaultdict

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
    # read data from excel files
    file_name = 'Datasett_NO1_Cleaned_r4.xlsx'
    data = inputData(file_name)

    # DCOPF_Model(data) #, b_matrix)
    return()



def inputData(file):
    # read data from file
    data = {}
    excel_sheets = ['Producers', 'Consumers', 'Time_wind', 'Node']
    for sheet in excel_sheets:
        df = pd.read_excel(file, sheet_name=sheet)
        data[sheet] = df.to_dict(orient='list')
    return data



def modelSetup(data):# , b_matrix):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    m = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    m.Producers = pyo.Set(initialize=xxx)  # Generators
    m.Consumers = pyo.Set(initialize=xxx)  # Load units
    m.Nodes = pyo.Set(initialize=xxx)  # Nodes

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


