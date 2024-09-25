import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Constants: dictionary storing maximum constraints for acres and sugar sales
constants = {'max_acres': 500,
             'max_sugar_sale': 6000}


# Function to read and structure input data from an Excel file
def InputData(data_file):
    # Read data from Excel and set 'Parameter' column as index
    inputdata = pd.read_excel(f'{data_file}')
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()  # Transpose to match expected format

    # Store sell and buy-related data separately in the 'data' dictionary
    data = {}
    data['sell'] = inputdata[['Price_sell', 'H_yield', 'Plant_cost']]  # Sell data includes price, yield, and cost
    data['buy'] = inputdata[['Price_buy', 'Min_req']].drop(
        'Sugar')  # Buy data includes price and minimum requirements, excluding sugar

    return data


# Read input data from the Excel file 'Farmers.xlsx'
data = InputData('Farmers.xlsx')


# Mathematical objective function
def Obj(m):
    # Objective is to maximize profit: revenue from selling crops minus planting costs and purchasing costs
    return sum(m.Ps[i] * m.w[i] - m.C[i] * m.x[i] for i in m.I) - sum(m.Pb[j] * m.y[j] for j in m.J)


# Constraint: ensure minimum production requirements are met
def MinReq(m, j):
    return m.H[j] * m.x[j] + m.y[j] - m.w[j] >= m.B[j]


# Constraint: limit on maximum sugar sale based on external market constraint
def MaxSugarSale(m):
    return m.w['Sugar'] <= m.MS


# Constraint: limit sugar sale to be less than or equal to sugar yield
def MaxSugarYield(m):
    return m.w['Sugar'] <= m.H['Sugar'] * m.x['Sugar']


# Constraint: total land used for planting crops should not exceed maximum available acres
def LandRestriction(m):
    return sum(m.x[i] for i in m.I) <= m.ML


# Function to set up and define the optimization model
def ModelSetUp(data, constants):
    # Create a Pyomo concrete model instance
    m = pyo.ConcreteModel()

    # Define sets: 'I' for crops sold and 'J' for crops bought
    m.I = pyo.Set(initialize=list(data['sell'].index))
    m.J = pyo.Set(initialize=list(data['buy'].index))

    # Define parameters for the model based on input data and constants
    m.Ps = pyo.Param(m.I, initialize=data['sell']['Price_sell'])  # Selling prices of crops
    m.Pb = pyo.Param(m.J, initialize=data['buy']['Price_buy'])  # Buying prices of crops
    m.H = pyo.Param(m.I, initialize=data['sell']['H_yield'])  # Yield for each crop
    m.C = pyo.Param(m.I, initialize=data['sell']['Plant_cost'])  # Planting costs for each crop
    m.B = pyo.Param(m.J, initialize=data['buy']['Min_req'])  # Minimum requirements for each crop to meet demand
    m.MS = pyo.Param(initialize=constants['max_sugar_sale'])  # Maximum allowable sugar sale
    m.ML = pyo.Param(initialize=constants['max_acres'])  # Maximum available land for planting

    # Define decision variables for land allocation (x), selling quantities (w), and buying quantities (y)
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)  # Land used for each crop (non-negative)
    m.w = pyo.Var(m.I, within=pyo.NonNegativeReals)  # Amount sold of each crop
    m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)  # Amount bought of each crop

    # Add constraints to the model
    m.MinReq = pyo.Constraint(m.J, rule=MinReq)  # Ensure minimum production requirement is satisfied
    m.MaxSugarSale = pyo.Constraint(rule=MaxSugarSale)  # Limit on sugar sale
    m.MaxSugarYield = pyo.Constraint(rule=MaxSugarYield)  # Limit on sugar yield for sale
    m.LandRestriction = pyo.Constraint(rule=LandRestriction)  # Restriction on total land use

    # Set the objective function: maximize profit
    m.obj = pyo.Objective(rule=Obj, sense=pyo.maximize)

    return m  # Return the constructed model


# Function to solve the optimization model using the Gurobi solver
def Solve(m):
    opt = SolverFactory("gurobi")  # Specify Gurobi as the solver
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)  # Enable dual variables to be captured for constraints
    results = opt.solve(m, load_solutions=True)  # Solve the model and load the results
    return results, m


# Function to display results including variable values and dual values
def DisplayResults(m):
    return print(m.display(), m.dual.display())  # Print model variables and dual variables


# Main process: set up the model, solve it, and display the results
m = ModelSetUp(data, constants)
Solve(m)
DisplayResults(m)
