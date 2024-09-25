from pyomo.environ import *
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# def inputData(file):
#     """
#     Reads data from the Excel file and returns a dictionary containing data from relevant sheets.
#
#     Parameters:
#     file (str): The path to the Excel file.
#
#     Returns:
#     dict: A dictionary containing data from each sheet as lists of dictionaries.
#     """
#     data = {}
#     excel_sheets = ['Producers', 'Consumers', 'Time_wind']
#     for sheet in excel_sheets:
#         df = pd.read_excel(file, sheet_name=sheet)
#         df.index += 1
#         data[sheet] = df.to_dict()
#     return data
#
#
#
# file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
# data = inputData(file_name)


model = ConcreteModel()

wind_scenario_values = {
    ('wind_med', '02/06/2020'): 20.32,
    ('wind_med', '03/06/2020'): 45.6,
    ('wind_high', '02/06/2020'): 20.32,
    ('wind_high', '03/06/2020'): 55.904,
    ('wind_low', '02/06/2020'): 20.32,
    ('wind_low', '03/06/2020'): 34.504
}

# Sets - Strukturerer indekser
model.Generators = Set(initialize=["nuclear", "hydro", "wind"])                                                         # Generators
model.Loads = Set(initialize=["Load 1"])                                                                                # Loads
model.Scenarios = Set(initialize=["wind_med", "wind_high", "wind_low"])                                                 # Wind scenarios
model.Time = Set(initialize=["02/06/2020", "03/06/2020"])                                                               # Time periods



# Parameters - Definerer konstanter/ verdier knyttet til settene. "Tenker på hvert sett som et objekt og så henger jeg verdier på dette objektet."
model.P_max = Param(model.Generators, initialize={"nuclear": 200, "hydro": 40, "wind": 80})                     # Her har jeg lagt inn 40 på hydro siden det er det som maks kan allokeres i day-ahead market.
model.P_min = Param(model.Generators, initialize={"nuclear": 0, "hydro": 0, "wind": 0})
model.MC = Param(model.Generators, initialize={"nuclear": 15, "hydro": 30, "wind": 0})
model.P_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 20, "wind": 0})                   # Reserved capacity for each generator for the next day [MW]
model.C_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 30, "wind": 0})                   # Cost for reserving capacity for the next day [Euro/MWh]
model.Load_Demand = Param(model.Loads, initialize={"Load 1": 250})                                              # Load demand for the inflexible load [MW]
model.C_rationing = Param(initialize=250)                                                                               # Rationing cost [Euro/MWh]
model.P_wind = Param(model.Scenarios, model.Time, initialize=wind_scenario_values)


#  Variables
model.Production = Var(model.Generators, model.Time, model.Scenarios, within=NonNegativeReals)                  # Produksjon fra nuclear og hydro
model.Rationing = Var(model.Loads, model.Time, model.Scenarios, within=NonNegativeReals)                        # Rasjonering (slakkvariabel for underdekning) Denne slakkvariabelen vil bidra til objektivfunksjonen med en høy kostnad, men vil bare bli aktiv dersom etterspørselen ikke kan møtes.



#  Constraints
def prod_lim_max(model, g, t, s):
    return model.Production[g, t, s] <= model.P_max[g]                                                                  # model.Production[g, t, s]: Beslutningsvariabelen som representerer hvor mye generator g produserer i tidsperioden t og scenarioet s.
model.ProdMaxCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_max)

def prod_lim_min(model, g, t, s):
    return model.Production[g, t, s] >= model.P_min[g]
model.ProdMinCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_min)

def wind_prod_limit(model, t, s):
    return model.Production['wind', t, s] <= model.P_wind[s, t]
model.WindProdLimit = Constraint(model.Time, model.Scenarios, rule=wind_prod_limit)                             # Begrenser produksjonen fra vind til det som er tilgjengelig i hvert scenario.

def reserved_cap_cons(model, t, s):
    return model.Production["hydro", t, s] >= model.P_reserved["hydro"]
model.ReservedCapacityCons = Constraint(model.Time, model.Scenarios, rule=reserved_cap_cons)

def load_balance_cons(model, l, t, s):
    return sum(model.Production[g, t, s] for g in model.Generators) + model.Rationing[l, t, s] >= model.Load_Demand[l]  # i model.Load_Demand[l] er l lasten. Trenger ikke t eller s fordi lasten er konstant over tid og scenario.
model.LoadBalanceCons = Constraint(model.Loads, model.Time, model.Scenarios, rule=load_balance_cons)


# Objectiv function
def objective_rule(model):
    production_cost = sum(model.Production[g, t, s] * model.MC[g] for g in model.Generators for t in model.Time for s in model.Scenarios)           # Produksjonskostnader
    reserved_cost = sum(model.P_reserved["hydro"] * model.C_reserved["hydro"] for t in model.Time)                                                  # Kostnader for å reservere hydro
    rationing_cost = sum(model.Rationing[l, t, s] * model.C_rationing for l in model.Loads for t in model.Time for s in model.Scenarios)            # Kostnader for rasjonering
    return production_cost + reserved_cost + rationing_cost                                                                                         # Total kostnad
model.Objective = Objective(rule=objective_rule, sense=minimize)


model.dual = Suffix(direction=Suffix.IMPORT)

solver = SolverFactory('glpk')
solver.solve(model, tee=True)
# model.Production.display()
# model.Rationing.display()


# Initialize an empty list to collect the results
results = []

# Loop through generators, time, and scenarios to collect production values
for t in model.Time:
    for s in model.Scenarios:
        nuclear_prod = model.Production['nuclear', t, s].value
        hydro_prod = model.Production['hydro', t, s].value
        wind_prod = model.Production['wind', t, s].value
        rationing_value = model.Rationing['Load 1', t, s].value
        # Append the results for each time, scenario, and generator to the list
        results.append({
            'Time': t,
            'Scenario': s,
            'Nuclear Production (MW)': nuclear_prod,
            'Hydro Production (MW)': hydro_prod,
            'Wind Production (MW)': wind_prod,
            'Rationing (MW)': rationing_value
        })

# Convert the list of results to a DataFrame for better presentation
df_results = pd.DataFrame(results)

# Juster Pandas visningsinnstillinger for å vise hele DataFrame
pd.set_option('display.max_columns', None)  # Vis alle kolonner
pd.set_option('display.expand_frame_repr', False)  # Ikke avkort kolonner

# Skriv ut DataFrame på nytt
print(df_results)

