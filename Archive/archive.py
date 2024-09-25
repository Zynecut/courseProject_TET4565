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


# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
# from pyomo.environ import *


# Structure

"""
Stochasticity in linear programming

Hard constraints -> Robust optimization
Soft constraints -> Chance constraints


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
    return sum(m.mc[g]*m.genHydro[g] + m.mc[g]*m.genWind[g] for g in m.Producers)

# Define probabilities for the three scenarios (low, med, high)
scenario_probabilities = {'low': 1/3, 'med': 1/3, 'high': 1/3}
# def StochasticObjFunction(m):
#     return sum(scenario_probabilities[s] * sum(m.mc[g] * m.gen[g] for g in m.Producers) for s in m.Scenarios)


def StochasticObjFunction(m):
    # Deterministic stage (no scenario)
    det_part = sum(m.mc[g] * (m.genHydro[g] + m.genWind[g]) for g in m.Producers)

    # Stochastic stage (wind scenarios only affect genWind)
    sto_part = sum(
        scenario_probabilities[sc] * sum(m.mc[g] * (m.genHydro[g] + m.genWind[g]) for g in m.Producers) for sc in
        m.Scenarios)

    return det_part + sto_part


# def GenerationLimit(m, g):
#    return m.pmin[g], m.gen[g], m.pmax[g]

# Load constraints (matching consumer demand)
def LoadLimit(m, l):
    for g in m.Producers:
        if m.source[g] == 'hydro':
            hydro = m.genHydro[g]
        else:
            wind = m.genWind[g]
    return hydro + wind >= m.Demand[l]
    # return sum(m.genHydro[g] + m.genWind[g] for g in m.Producers) >= m.Demand[l]
    # return sum(m.genHydro[g] + m.genWind[g] for g in m.Producers) + \
    #       sum(m.genWind[g] for g in m.Producers if m.source[g] == 'wind') >= m.Demand[l]


def HydroGenerationLimit(m, g):
    if m.source[g] == 'hydro':
        return m.pmin[g], m.genHydro[g], m.pmax[g]
    else:
        return pyo.Constraint.Skip  # Skip for non-hydro generators


# Deterministic wind generation for the first two days
def WindGenerationDeterministic(m, g, s):
    if s == 'det' and m.source[g] == 'wind':  # For the first two days
        return m.pmin[g], m.genWind[g], m.pmax[g]*m.wind_avg
    else:
        return pyo.Constraint.Skip  # Skip for the third day (handled by scenarios)


"""
Det som ikke funker her er at gen er jo mye større enn pmax, siden den allerede har dratt inn hydro??
"""


# Stochastic wind generation for the third day
def WindGenerationStochastic(m, g, s, sc):
    if s == 'sto' and m.source[g] == 'wind':  # For the third day (stochastic period)
        if sc == 'low':
            return m.pmin[g], m.genWind[g], m.pmax[g]*m.wind_avg_low
        elif sc == 'med':
            return m.pmin[g], m.genWind[g], m.pmax[g]*m.wind_avg_med
        elif sc == 'high':
            return m.pmin[g], m.genWind[g], m.pmax[g]*m.wind_avg_high
    else:
        return pyo.Constraint.Skip


def NonAnticipativityConstraint(m, g):
    # Ensure that wind generation in the deterministic stage is consistent across scenarios in the stochastic stage
    return m.genWind[g] == m.genWind[g]  # Only for wind, no need for hydro



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
    m.Stage = pyo.Set(initialize=['det', 'sto'])  # Stages
    m.Scenarios = pyo.Set(initialize=['low', 'med', 'high'])  # Scenarios

    """
    Parameters
    """
    # Generator data
    m.pmax          = pyo.Param(m.Producers, initialize=data['Producers']['pmax'])          # Max power output
    m.pmin          = pyo.Param(m.Producers, initialize=data['Producers']['pmin'])          # Min power output
    m.mc            = pyo.Param(m.Producers, initialize=data['Producers']['marginal_cost']) # Marginal cost
    m.storage_cap   = pyo.Param(m.Producers, initialize=data['Producers']['storage_cap'])   # Storage capacity
    m.source        = pyo.Param(m.Producers, initialize=data['Producers']['gen_source'])    # Type of generator

    # Load data
    m.Demand        = pyo.Param(m.Consumers, initialize=data['Consumers']['consumption'])       # Demand
    m.Rationing     = pyo.Param(m.Consumers, initialize=data['Consumers']['Rationing cost'])    # Rationing

    # Wind data
    # Deterministic wind generation for the first two days
    m.wind_avg = pyo.Param(initialize=average_wind(data['Time_wind']['wind_med'], [1, 2, 3, 4, 5, 6, 7, 8]))

    # Scenarios of stochastic wind generation for the third day
    m.wind_avg_low = pyo.Param(initialize=average_wind(data['Time_wind']['wind_low'], [9, 10, 11, 12, 13]))
    m.wind_avg_med = pyo.Param(initialize=average_wind(data['Time_wind']['wind_med'], [9, 10, 11, 12, 13]))
    m.wind_avg_high = pyo.Param(initialize=average_wind(data['Time_wind']['wind_high'], [9, 10, 11, 12, 13]))

    """
    Variables
    """
    # m.gen = pyo.Var(m.Producers, within=pyo.NonNegativeReals)  # Power output
    m.genHydro = pyo.Var(m.Producers, within=pyo.NonNegativeReals)
    m.genWind = pyo.Var(m.Producers, within=pyo.NonNegativeReals)

    """
    Constraints
    """
    # m.GenLimit_Const = pyo.Constraint(m.Producers, rule=GenerationLimit)    # Power output constraint

    # Hydro generation (no stochastic behavior)
    m.HydroGen_Const = pyo.Constraint(m.Producers, rule=HydroGenerationLimit)

    # Deterministic wind generation for the first two days
    m.WindGen_Det = pyo.Constraint(m.Producers, m.Stage, rule=WindGenerationDeterministic)

    # Stochastic wind generation for the third day
    m.WindGen_Sto = pyo.Constraint(m.Producers, m.Stage, m.Scenarios, rule=WindGenerationStochastic)

    # m.NonAnticipativity_Constraint = pyo.Constraint(m.Producers, rule=NonAnticipativityConstraint)
    """
    Må slå sammen Det og Sto??
    """

    # Load constraint
    m.Load_Const = pyo.Constraint(m.Consumers, rule=LoadLimit)              # Load constraint

    """
    Objective Function
    """
    # Define objective function
    # m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)
    m.obj = pyo.Objective(rule=StochasticObjFunction, sense=pyo.minimize)

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


