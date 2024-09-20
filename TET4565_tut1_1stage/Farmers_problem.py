import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Constants
constants = {'max_acres':500,
             'max_sugar_sale':6000}

def InputData(data_file):
    inputdata = pd.read_excel(f'{data_file}')
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()
    data = {}
    data['sell'] = inputdata[['Price_sell', 'H_yield', 'Plant_cost']]
    data['buy'] = inputdata[['Price_buy', 'Min_req']].drop('Sugar')
    return data
data = InputData('Farmers.xlsx')


# Mathematical formulation
def Obj(m):
    return sum(m.Ps[i]*m.w[i] - m.C[i]*m.x[i] for i in m.I) - sum(m.Pb[j]*m.y[j] for j in m.J)
def MinReq(m, j):
    return m.H[j]*m.x[j] + m.y[j] - m.w[j] >= m.B[j]
def MaxSugarSale(m):
    return m.w['Sugar'] <= m.MS
def MaxSugarYield(m):
    return m.w['Sugar'] <= m.H['Sugar'] * m.x['Sugar']
def LandRestriction(m):
    return sum(m.x[i] for i in m.I) <= m.ML


# Set up model
def ModelSetUp(data, constants):
    # Instance
    m = pyo.ConcreteModel()

    # Define sets
    m.I = pyo.Set(initialize=list(data['sell'].index))
    m.J = pyo.Set(initialize=list(data['buy'].index))

    # Define parameters
    m.Ps = pyo.Param(m.I, initialize=data['sell']['Price_sell'])
    m.Pb = pyo.Param(m.J, initialize=data['buy']['Price_buy'])
    m.H = pyo.Param(m.I, initialize=data['sell']['H_yield'])
    m.C = pyo.Param(m.I, initialize=data['sell']['Plant_cost'])
    m.B = pyo.Param(m.J, initialize=data['buy']['Min_req'])
    m.MS = pyo.Param(initialize=constants['max_sugar_sale'])
    m.ML = pyo.Param(initialize=constants['max_acres'])

    # Define variables
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.w = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)

    # Define constraints
    m.MinReq = pyo.Constraint(m.J, rule=MinReq)
    m.MaxSugarSale = pyo.Constraint(rule=MaxSugarSale)
    m.MaxSugarYield = pyo.Constraint(rule=MaxSugarYield)
    m.LandRestriction = pyo.Constraint(rule=LandRestriction)

    # Define objective function
    m.obj = pyo.Objective(rule=Obj, sense=pyo.maximize)
    return m
def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m
def DisplayResults(m):
    return print(m.display(), m.dual.display())

m = ModelSetUp(data, constants)
Solve(m)
DisplayResults(m)

