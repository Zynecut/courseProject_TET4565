import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# Function to load data including wind data
def inputData(file):
    data = {}
    excel_sheets = ['Producers', 'Consumers', 'Time_wind', 'Node']
    for sheet in excel_sheets:
        df = pd.read_excel(file, sheet_name=sheet)
        data[sheet] = df.to_dict(orient='list')
    return data


# Use the inputData function to load data
data = inputData('Datasett_NO1_Cleaned_r4.xlsx')


# Objective function to minimize total cost of generation
def ObjectiveFunction(m):
    return sum(m.C[i] * m.x[i] for i in m.I)


# Constraints: generation limits with wind factor
def GenerationUpperLimit(m, i):
    # Find the index of the producer in the data
    index = data['Producers']['Producer'].index(i)

    # Check if the producer is wind-based
    if data['Producers']['type'][index] == 'wind':
        wind_ratio = data['Time_wind']['wind_med'][0]  # Use 'wind_med' or any wind data column
        return m.x[i] <= wind_ratio * m.pmax[i]
    else:
        return m.x[i] <= m.pmax[i]


def GenerationLowerLimit(m, i):
    return m.x[i] >= m.pmin[i]


# Load constraints (matching consumer demand)
def LoadConstraint(m, j):
    return sum(m.x[i] for i in m.I) >= m.demand[j]


# Model setup with wind production limits
def ModelSetUpMinCost(data):
    m = pyo.ConcreteModel()

    # Define sets
    producers = data['Producers']['Producer']
    consumers = data['Consumers']['nodeID']

    m.I = pyo.Set(initialize=producers)  # Producers
    m.J = pyo.Set(initialize=consumers)  # Consumers

    # Define parameters for producers
    pmax_dict = dict(zip(data['Producers']['Producer'], data['Producers']['pmax']))
    pmin_dict = dict(zip(data['Producers']['Producer'], data['Producers']['pmin']))
    marginal_cost_dict = dict(zip(data['Producers']['Producer'], data['Producers']['marginal_cost']))

    m.pmax = pyo.Param(m.I, initialize=pmax_dict)
    m.pmin = pyo.Param(m.I, initialize=pmin_dict)
    m.C = pyo.Param(m.I, initialize=marginal_cost_dict)

    # Define parameters for consumers (demand from consumption)
    demand_dict = dict(zip(data['Consumers']['nodeID'], data['Consumers']['consumption']))
    m.demand = pyo.Param(m.J, initialize=demand_dict, within=pyo.Reals)

    # Define variables for generation
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)

    # Define constraints
    m.GenUpperLimit = pyo.Constraint(m.I, rule=GenerationUpperLimit)
    m.GenLowerLimit = pyo.Constraint(m.I, rule=GenerationLowerLimit)

    # Load constraint: Ensure the sum of generation meets the load
    m.LoadConstraint = pyo.Constraint(m.J, rule=LoadConstraint)

    # Define objective function
    m.obj = pyo.Objective(rule=ObjectiveFunction, sense=pyo.minimize)

    return m


# Solve function
def SolveModel(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m


# Display results
def DisplayModelResults(m):
    return m.pprint()


# Model setup and solve
m = ModelSetUpMinCost(data)
results, m = SolveModel(m)
DisplayModelResults(m)
