# Benders decomposition for multiple scenarios
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import time


def main():
    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    # benders(data)
    start_time = time.time()
    SDP(data)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")

def inputData(file):
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


"""Master problem formulation"""
def Obj_1st(m):
    return m.nuclear_DA * m.MC['nuclear'] + m.hydro_DA * m.MC['hydro'] + m.hydro_res_DA * m.C_res + m.alpha
"""Constraints"""
def DA_load_balance(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand
def nuclear_lim(m):
    return m.P_min['nuclear'], m.nuclear_DA, m.P_max['nuclear']
def hydro_lim(m):
    return m.P_min['hydro'], m.hydro_DA + m.hydro_res_DA , m.P_max['hydro']
def hydro_res_min(m):
    return m.hydro_res_DA >= 0.01  # Minimum reserve for hydro
def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] - m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])


def masterModel(data, Cuts):
    """Setup Master Problem Model"""
    m = pyo.ConcreteModel()
    """Sets"""
    m.G             = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))
    """Parameters"""
    m.MC            = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.C_res         = pyo.Param(initialize=data['Producers']['reserve_cost']['hydro'])
    m.demand        = pyo.Param(initialize=data['Consumers']['consumption']['Load 1'])
    m.P_max         = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min         = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.wind_DA       = pyo.Param(initialize=data['Time_wind']['med'])
    """Variables"""
    m.nuclear_DA    = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_DA      = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, m.P_max['hydro']))
    m.hydro_res_DA  = pyo.Var(within=pyo.NonNegativeReals)
    m.alpha         = pyo.Var(bounds=(-1000, 1000))
    """Cuts"""
    m.Cut           = pyo.Set(initialize=Cuts["Set"])  # Set for cuts
    m.Phi           = pyo.Param(m.Cut, initialize=Cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda        = pyo.Param(m.Cut, initialize=Cuts["lambda"])  # Parameter for lambda (dual value of reserve)
    m.X_hat         = pyo.Param(m.Cut, initialize=Cuts["x_hat"])  # Parameter for reserved hydro
    """Constraints"""
    m.DA_balance    = pyo.Constraint(rule=DA_load_balance)
    m.nuclear_lim   = pyo.Constraint(rule=nuclear_lim)
    m.hydro_lim     = pyo.Constraint(rule=hydro_lim)
    m.hydro_res_min = pyo.Constraint(rule=hydro_res_min)
    m.CreateCuts    = pyo.Constraint(m.Cut, rule=CreateCuts)
    """Objective Function"""
    m.obj           = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)
    return m


"""Sub-problem formulation"""
def Obj_2nd(m):
    return sum(m.prob[s] * ((m.hydro_RT[s]-m.hydro_DA) * m.MC['hydro'] + m.rationing[s] * m.C_rat) for s in m.S)
"""Constraints"""
def RT_load_balance(m, s):
    return m.hydro_RT[s] + m.wind_RT[s] + m.nuclear_RT[s] + m.rationing[s] >= m.demand
def hydro_link_RT(m, s):
    """Link upper bound for hydro_RT with hydro_DA + hydro_res_DA"""
    return m.hydro_RT[s] <= m.hydro_DA + m.X_hat
def hydro_lower_RT(m, s):
    """Link lower bound for hydro_RT: hydro_RT >= hydro_DA - hydro_res_DA"""
    return m.hydro_RT[s] >= m.hydro_DA - m.X_hat

def subModel(data, X_hat, DA_values, probability):
    """Setup Sub Problem Model"""
    m = pyo.ConcreteModel()
    """Sets"""
    m.G             = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))
    m.S             = pyo.Set(initialize=list(data['Time_wind'].keys()))
    """Parameters"""
    m.MC            = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.C_rat         = pyo.Param(initialize=data['Consumers']['rationing_cost']["Load 1"])
    m.demand        = pyo.Param(initialize=data['Consumers']['consumption']['Load 1'])
    m.nuclear_RT    = pyo.Param(m.S, initialize=DA_values["nuclear_DA"])
    m.wind_RT       = pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob          = pyo.Param(m.S, initialize=probability)
    m.hydro_DA      = pyo.Param(initialize=DA_values["hydro_DA"])
    m.X_hat         = pyo.Param(initialize=X_hat)
    """Variables"""
    m.hydro_RT      = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.rationing     = pyo.Var(m.S, bounds=(0, 250), within=pyo.NonNegativeReals)
    """Constraints"""
    m.RT_balance    = pyo.Constraint(m.S, rule=RT_load_balance)
    m.hydro_high_RT = pyo.Constraint(m.S, rule=hydro_link_RT)
    m.hydro_low_RT  = pyo.Constraint(m.S, rule=hydro_lower_RT)
    """Objective Function"""
    m.obj           = pyo.Objective(rule=Obj_2nd, sense=pyo.minimize)
    return m


def manageCuts(Cuts, m):
    """Add new cut to existing dictionary of cut information"""
    cut = len(Cuts["Set"])
    Cuts['Set'].append(cut)
    Cuts['Phi'][cut] = pyo.value(m.obj)
    Cuts['lambda'][cut] = sum(m.dual[m.RT_balance[s]]for s in m.S)  # Retrieve duals for each scenario
    Cuts['x_hat'][cut] = pyo.value(m.X_hat)
    return Cuts


def SDP(data):
    """Setup for Stochastic Dynamic Programming (SDP"""

    """Setup for Stochastic dynamic programming - Data curation"""

    # Pre-step: determine the discretization we want to explore in the second-stage
    Min = 0
    Max = 5.5
    # How large each discrete jump is in value
    states_jump = 2.6  # 10 values: 0.5777, 3 values: 2.6
    # List_states = [i for i in range(Min, Max, states_jump)]
    List_states = [i for i in np.arange(Min, Max, states_jump)]

    # Define the list of initial values for each decision variable
    Reserved_initial_value = List_states

    """
    Itertools is a package that can be used to create sophisticated lists with tuple-elements.
    Itertools.product creates a list of combinations for the given input of lists, as tuples.
    """
    # This package can deal with creating combinations of multiple lists
    import itertools

    # We create a list of tuples that contain all combinations of combined occurences of each element in the three lists
    # List_combinations = [p for p in itertools.product(Reserved_initial_value)]
    # Each tuple will contain the initial value for each type of grain. Order is important: 1st tuple is wheat, 2nd is corn, 3rd is sugar (as indicated in line above)

    """Start the SDP process"""

    # Pre-step: Formulate cut input data
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    x_values = []
    alpha_values = []

    # For each combination we acquired
    for initial_value in Reserved_initial_value:
        # Set 1st stage result
        X_hat = initial_value
        DA_values = {"nuclear_DA": 150, "hydro_DA": 54.80}
        probability = {'low': 1, 'med':0 , 'high': 0}


        # If the combination is invalid (sum of grain planted > Max), we skip
        # If within allowed limits
        if X_hat <= Max:
            # Solve 2nd stage, store cuts
            m_2nd = subModel(data, X_hat, DA_values, probability)
            Solve(m_2nd)


            Cuts = manageCuts(Cuts, m_2nd)
        # If planted is higher than allowed

            x_values.append(X_hat)
            alpha_values.append(pyo.value(m_2nd.obj))
        else:
            pass

    # Solve the 1st stage problem with the acquired cuts
    m_1st = masterModel(data, Cuts)
    Solve(m_1st)

    X_hat = pyo.value(m_1st.hydro_res_DA)

    # Print results 1st stage
    print(f"X_hat (Hydro Reserve DA): {X_hat:.2f}")
    print(pyo.value(m_1st.alpha.value))
    print(f"Objective Value: {pyo.value(m_1st.obj)}")
    # print(Cuts)
    # Plotting av cut generering
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, alpha_values, 'o-', color='teal', label="Cuts")
    for i, (x, y) in enumerate(zip(x_values, alpha_values)):
        plt.annotate(f"Cut {i+1}: ({x:.2f}, {y:.2f})", (x, y), ha='right', va='bottom')
    plt.xlabel("Hydro Reserve DA [MW]")
    plt.ylabel("Objective Value (Second-Stage)")
    plt.title("Cut Generation in SDP")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting av Objective cost for different cuts
    # plt.figure(figsize=(10, 6))
    # plt.plot(Cuts['Set'], Cuts['Phi'].values(), 'o-', color='teal', label="Cuts")
    # plt.xlabel("Cuts")
    # plt.ylabel("Objective Value for different cuts")
    # plt.title("Objective value for each cut")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m


if __name__ == '__main__':
    main()