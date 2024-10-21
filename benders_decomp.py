# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
from sphinx.ext.graphviz import graphviz


def main():
    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    benders(data)


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

    df_prod = pd.DataFrame(data['Producers'])
    df_prod = df_prod.set_index('type')
    data['Producers'] = df_prod.to_dict()

    df_load = pd.DataFrame(data['Consumers'])
    df_load = df_load.set_index('load')
    data['Consumers'] = df_load.to_dict()

    df = pd.DataFrame(data['Time_wind'])
    df = df.set_index('Stage')
    df = df.reset_index(drop=True)
    data['Time_wind'] = df.iloc[0].to_dict()
    return data

const = {'MC_res': 25}

"""Mathematical formulation 1st stage"""
def Obj_1st(m):
    production_cost_DA = m.hydro_res_DA * const['MC_res'] + m.hydro_DA * m.MC['hydro'] + m.nuclear_DA * m.MC['nuclear']
    return production_cost_DA + m.alpha


"""Set Variable Bounds"""
# Nuclear DA bounds
def bounds_nuclear_DA(m):
    return m.P_min["nuclear"], m.P_max["nuclear"]  # Begrens produksjonen til maksimal produksjon for kjernekraft.
# Hydro DA bounds
def bounds_hydro_DA(m):
    return m.P_min["hydro"], m.P_max["hydro"]
# Load Balance DA
def load_balance_DA(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand["Load 1"]
# Hydro Reserve Min
def hydro_res_min(m):
    return m.hydro_res_DA >= 0
# Creating cuts
def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] + m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])


"""Mathematical formulation 2nd stage"""
def Obj_2st(m):
    production_cost_RT = sum(m.prob[s] * (m.MC['hydro'] * (m.hydro_RT[s] - m.hydro_DA)) for s in m.S)
    rationing_cost = sum(m.prob[s] * m.cost_rat[l] * m.rationing[l, s] for l in m.L for s in m.S)
    return production_cost_RT + rationing_cost
# Load Balance RT
def load_balance_RT(m, s):
    return m.hydro_RT[s] + m.P_wind[s] + m.nuclear_RT + m.rationing["Load 1", s] >= m.demand["Load 1"]
def rationing_limits(m, l, s):
    return 0, m.demand[l]
def hydro_upper_RT(m, s):
    return m.hydro_RT[s] <= m.hydro_DA + m.X_hat
def hydro_lower_RT(m, s):
    return m.hydro_RT[s] >= m.hydro_DA - m.X_hat



def Model_1st(data, cuts):
    m = pyo.ConcreteModel()
    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')
    """Parameters"""
    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.wind_DA = pyo.Param(initialize=50)
    """Variables"""
    m.nuclear_DA = pyo.Var(bounds=bounds_nuclear_DA, within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(bounds=bounds_hydro_DA, within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    """Cuts"""
    m.Cut = pyo.Set(initialize=cuts["Set"])  # Set for cuts
    m.Phi = pyo.Param(m.Cut, initialize=cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda = pyo.Param(m.Cut, initialize=cuts["lambda"])  # Parameter for lambda (dual value of reserve)
    m.X_hat = pyo.Param(m.Cut, initialize=cuts["x_hat"])  # Parameter for reserve level
    # Variable for alpha
    m.alpha = pyo.Var(bounds=(-100000, 100000))  # Variable for alpha
    # Cut constraint
    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)
    """Constraints"""
    m.LoadBalance_DA = pyo.Constraint(rule=load_balance_DA)
    m.HydroResMin = pyo.Constraint(rule=hydro_res_min)
    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)
    return m


def Model_2nd(data, X_hat, DA_values):
    m = pyo.ConcreteModel()
    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')
    m.S = pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')
    """Parameters"""
    # m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    # m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.cost_rat = pyo.Param(m.L, initialize=data['Consumers']['rationing_cost'])
    m.P_wind = pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob = pyo.Param(m.S, initialize={'low': 1 / 3, 'med': 1 / 3, 'high': 1 / 3})
    m.X_hat = pyo.Param(initialize=X_hat)
    m.nuclear_RT = pyo.Param(initialize=DA_values["nuclear_DA"])
    m.hydro_DA = pyo.Param(initialize=DA_values["hydro_DA"])
    """Variables"""
    m.hydro_RT = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.rationing = pyo.Var(m.L, m.S, bounds= rationing_limits, within=pyo.NonNegativeReals)
    """Constraints"""
    # m.LockedNuclearProd = pyo.Constraint(m.S, rule=locked_nuclear_prod)
    m.LoadBalance_RT = pyo.Constraint(m.S, rule=load_balance_RT)
    m.HydroUpper_RT = pyo.Constraint(m.S, rule=hydro_upper_RT)  # Hydro_RT <= Hydro_DA + reserved capacity
    m.HydroLower_RT = pyo.Constraint(m.S, rule=hydro_lower_RT)
    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_2st, sense=pyo.minimize)
    return m


def manage_Cuts(Cuts, cut_info):
    """Add new cut to existing dictionary of cut information"""

    # Find cut iteration by checking number of existing cuts
    cut = len(Cuts["Set"])
    # Add new cut to list, since 0-index is a thing that works well
    Cuts["Set"].append(cut)

    # Find 2nd stage cost result
    Cuts["Phi"][cut]    = cut_info["Phi"]
    Cuts["lambda"][cut] = cut_info["lambda"]
    Cuts["x_hat"][cut]  = cut_info["x_hat"]
    return Cuts



def benders(data):
    # Initial cuts and parameters
    Cuts = {"Set": [], "Phi": {}, "lambda": {}, "x_hat": {}}
    graph  = {"UB": {}, "LB": {}}

    # iteration loop
    for i in range(1, 3):
        # First-stage model (master problem)
        m_1st = Model_1st(data, Cuts)
        SolveModel(m_1st)

        # Save X_hat (hydro_res_DA verdi) from first stage solution
        X_hat = m_1st.hydro_res_DA


        DA_values = {"nuclear_DA": pyo.value(m_1st.nuclear_DA), "hydro_DA": pyo.value(m_1st.hydro_DA)}

        print(f"Iteration: {i}")
        print(f"X_hat: {X_hat.value}")
        print(f"DA_values: {DA_values}")

        # Second-stage solution (sub-problem)
        # wind_scenario = {}  # Put wind scenario here
        cut_info = {"Phi": 0, "lambda": 0, "x_hat": 0}
        for s in data['Time_wind'].keys():
            m_2nd = Model_2nd(data, X_hat, DA_values)
            SolveModel(m_2nd)
            cut_info["Phi"]     += pyo.value(m_2nd.obj)
            cut_info["lambda"]  += pyo.value(m_2nd.dual[m_2nd.LoadBalance_RT[s]])
            cut_info["x_hat"]   += pyo.value(m_2nd.hydro_RT[s])

        Cuts = manage_Cuts(Cuts, cut_info)
        # Print 2nd stage results
        print("Objective function: ", pyo.value(m_2nd.obj))
        print("Cut information: ")

        for comp in Cuts:
            if comp == "lambda" or comp == "x_hat":
                print(comp, Cuts[comp])
            else:
                print(comp, Cuts[comp])

        graph["UB"][i] = pyo.value(m_1st.alpha.value)
        graph["LB"][i] = pyo.value(cut_info["Phi"])
        "Convergence check"
        print("UB:", pyo.value(m_1st.alpha.value), "- LB:", pyo.value(cut_info["Phi"]))
        # if (abs(pyo.value(m_1st.alpha.value) - pyo.value(cut_info['Phi'])) <= 0.001) or i > 10:


        # m_2nd = Model_2nd(data, X_hat, DA_values)
        # SolveModel(m_2nd)

        # Get dual value for LoadBalance_RT constraint
        # dual_lambda = m_2nd.dual[m_2nd.LoadBalance_RT]

        # Generate cuts based on dual values
        # Update cuts-sets with values
        # Cuts["Set"].append(i)
        # Cuts["Phi"][i] = ...  # Put correct Phi based on sub-problem result
        # Cuts["lambda"][i] = dual_lambda
        # Cuts["x_hat"][i] = X_hat

    DisplayModelResults(m_1st)



# Solve function
def SolveModel(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    # results = opt.solve(m, options={"ResultFile": "infeasibility.ilp"})
    return results, m

def DisplayModelResults(m):
    # return m.pprint()
    return print(m.display(), m.dual.display())


if __name__ == '__main__':
    main()