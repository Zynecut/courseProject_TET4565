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
    file_name = 'Datasett_NO1_Cleaned_r2.xlsx'
    data = read_data(file_name)
    # b_matrix = create_B_matrix(data)
    data['Time'] = data_preprocessing(data['Time'])
    DCOPF_Model(data) #, b_matrix)
    return()


def read_data(file):
    # read data from file
    data = {}
    excel_sheets = ['Producers', 'Consumers', 'Time', 'Node']
    for sheet in excel_sheets:
        df = pd.read_excel(file, sheet_name=sheet)
        data[sheet] = df.to_dict(orient='list')
    return data


def create_B_matrix(data):
    length = len(data['Node'])
    b_matrix = np.zeros((length, length))
    return b_matrix

def data_preprocessing(data):
    # Preprocess data
    # Convert time strings to datetime objects and categorize by day/night
    processed_data = {}
    n = 0
    for time in data['referenceTime']:
        processed_data[n] = {
            'referenceTime': time,
            'Period': 'Day' if 6 <= time.hour < 18 else 'Night',
            'Date': time.date()
        }
        n += 1

    data['referenceTime'] = processed_data

    # Initialize dictionaries to hold summed values and counts
    day_dict = defaultdict(lambda: {'load_NO1': 0, 'windon_NO1': 0, 'solar_NO1': 0, 'count': 0})
    night_dict = defaultdict(lambda: {'load_NO1': 0, 'windon_NO1': 0, 'solar_NO1': 0, 'count': 0})

    # Sum up the values for day and night periods
    n = 0
    for entry in data:
        if entry['referenceTime'][n]['Period'] == 'Day':
            day_dict[entry['Date']]['load_NO1'] += entry['load_NO1']
            day_dict[entry['Date']]['windon_NO1'] += entry['windon_NO1']
            day_dict[entry['Date']]['solar_NO1'] += entry['solar_NO1']
            day_dict[entry['Date']]['count'] += 1
            n += 1
        else:
            night_dict[entry['Date']]['load_NO1'] += entry['load_NO1']
            night_dict[entry['Date']]['windon_NO1'] += entry['windon_NO1']
            night_dict[entry['Date']]['solar_NO1'] += entry['solar_NO1']
            night_dict[entry['Date']]['count'] += 1
            n += 1

    # Calculate averages for each day
    final_data = {}
    for date in day_dict.keys():
        final_data[date] = {
            'load_NO1_Day': day_dict[date]['load_NO1'] / day_dict[date]['count'],
            'windon_NO1_Day': day_dict[date]['windon_NO1'] / day_dict[date]['count'],
            'solar_NO1_Day': day_dict[date]['solar_NO1'] / day_dict[date]['count'],
            'load_NO1_Night': night_dict[date]['load_NO1'] / night_dict[date]['count'],
            'windon_NO1_Night': night_dict[date]['windon_NO1'] / night_dict[date]['count'],
            'solar_NO1_Night': night_dict[date]['solar_NO1'] / night_dict[date]['count'],
        }

    # Convert the final_data dictionary to a DataFrame for easy viewing
    final_df = pd.DataFrame.from_dict(final_data, orient='index').reset_index().rename(columns={'index': 'Date'})

    return final_df

def DCOPF_Model(data):# , b_matrix):
    """
    Set up the optimization model, run it and store the data in a .xlsx file
    """
    model = pyo.ConcreteModel() #Establish the optimization model, as a concrete model in this case

    """
    Sets
    """
    model.Producers = pyo.Set(initialize=xxx)  # Generators
    model.Consumers = pyo.Set(initialize=xxx)  # Load units
    model.Nodes = pyo.Set(initialize=xxx)  # Nodes

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


