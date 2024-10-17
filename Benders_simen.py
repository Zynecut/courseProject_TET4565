# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# Structure
"""
Stochasticity in linear programming with Benders Decomposition
"""


def main():
    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    benders(data)


def inputData(file):
    """
    Reads data from the Excel file and returns a dictionary containing data from relevant sheets.
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

"""Master model - 1st stage"""


def Model_1st(data, Cuts):
    m = pyo.ConcreteModel()

    # Sets
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # Producers
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # Loads
    m.S = pyo.Set(initialize=list(data['Time_wind'].keys()))  # Scenarios

    # Parameters
    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.cost_rat = pyo.Param(m.L, initialize=data['Consumers']['rationing cost'])
    m.P_wind = pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob = pyo.Param(m.S, initialize={'low': 1 / 3, 'med': 1 / 3, 'high': 1 / 3})

    # Variables
    m.nuclear_DA = pyo.Var(bounds=(m.P_min['nuclear'], m.P_max['nuclear']), within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(bounds=(m.P_min['hydro'], m.P_max['hydro']), within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.wind_prod_DA = pyo.Var(within=pyo.NonNegativeReals)

    # Alpha variable (for cuts)
    m.alpha = pyo.Var(bounds=(-100000, 100000))

    # Cuts
    m.Cut = pyo.Set(initialize=Cuts["Set"])
    m.Phi = pyo.Param(m.Cut, initialize=Cuts["Phi"])
    m.Lambda = pyo.Param(m.Cut, initialize=Cuts["lambda"])
    m.X_hat = pyo.Param(m.Cut, initialize=Cuts["x_hat"])

    # Constraints
    m.LockedNuclearProd = pyo.Constraint(m.S, rule=lambda m, s: m.nuclear_DA == m.nuclear_DA)
    m.LoadBalance_DA = pyo.Constraint(rule=lambda m: m.nuclear_DA + m.hydro_DA + m.wind_prod_DA == m.demand["Load 1"])
    m.WindProdDAConstraint = pyo.Constraint(expr=m.wind_prod_DA == sum(m.prob[s] * m.P_wind[s] for s in m.S))

    # Add cuts as constraints
    def CreateCuts(m, c):
        return m.alpha >= m.Phi[c] + m.Lambda[c] * (m.hydro_DA - m.X_hat[c])

    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)

    # Objective function for first stage
    def Obj_1st(m):
        return m.hydro_res_DA * const['MC_res'] + m.alpha + m.hydro_DA * m.MC['hydro'] + m.nuclear_DA * m.MC['nuclear']

    m.Obj_1st = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)

    return m


"""Subproblem - 2nd stage"""


def Model_2nd(data, X_hat, wind):
    m = pyo.ConcreteModel()

    # Sets
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))
    m.S = pyo.Set(initialize=list(data['Time_wind'].keys()))  # Scenarios
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))

    # Parameters
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.P_wind = pyo.Param(m.S, initialize=data['Time_wind'])
    m.X_hat = pyo.Param(initialize=X_hat)
    m.wind = pyo.Param(initialize=wind)

    # Variables
    m.hydro_RT = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.rationing = pyo.Var(m.L, m.S, within=pyo.NonNegativeReals)

    # Constraints
    m.LoadBalance_RT = pyo.Constraint(m.S, rule=lambda m, s: m.hydro_RT[s] + m.wind >= m.demand["Load 1"])

    # Objective function for second stage
    def Obj_2nd(m):
        return sum(m.hydro_RT[s] * m.MC['hydro'] for s in m.S)

    m.Obj_2nd = pyo.Objective(rule=Obj_2nd, sense=pyo.minimize)

    return m


"""Function for adding cuts"""


def Cut_manage(Cuts, store):
    cut = len(Cuts["Set"])
    Cuts["Set"].append(cut)
    Cuts["Phi"][cut] = store['Phi']
    Cuts["lambda"][cut] = store['Lambda']
    Cuts["x_hat"][cut] = store['X_hat']
    return Cuts


"""Benders Decomposition"""


def benders(data):
    Cuts = {
        "Set": [],
        "Phi": {},
        "lambda": {},
        "x_hat": {}
    }

    graph = {"UB": {}, "LB": {}}
    i = 0
    run = True

    while run:
        print(f"\nIteration {i}:")
        # Solve first stage
        m_1st = Model_1st(data, Cuts)
        Solve(m_1st)

        # Print objective value of the first stage
        print(f"First Stage Objective: {pyo.value(m_1st.Obj_1st)}")

        # Get first stage solution and print variable values
        X_hat = m_1st.hydro_DA
        print(f"First Stage Hydro_DA: {X_hat.value}")
        print(f"Nuclear_DA: {m_1st.nuclear_DA.value}")

        # Solve second stage for each scenario
        store = {'Phi': 0, 'Lambda': 0, 'X_hat': 0}
        for s in data['Time_wind'].keys():
            m_2nd = Model_2nd(data, X_hat, data['Time_wind'][s])
            Solve(m_2nd)

            # Print objective value of the second stage for each scenario
            print(f"Second Stage Objective (Scenario {s}): {pyo.value(m_2nd.Obj_2nd)}")
            print(f"Hydro_RT (Scenario {s}): {m_2nd.hydro_RT[s].value}")

            # Aggregate results for cut management
            store['Phi'] += pyo.value(m_2nd.Obj_2nd) / len(data['Time_wind'].keys())
            store['Lambda'] += m_2nd.hydro_RT[s].value / len(data['Time_wind'].keys())
            store['X_hat'] += X_hat.value / len(data['Time_wind'].keys())

        # Add cuts to first stage
        Cuts = Cut_manage(Cuts, store)

        # Store UB and LB
        graph['UB'][i] = pyo.value(m_1st.Obj_1st)
        graph['LB'][i] = store['Phi']

        # Print UB and LB values
        print(f"Upper Bound (UB): {graph['UB'][i]}")
        print(f"Lower Bound (LB): {graph['LB'][i]}")

        # Check convergence
        if abs(graph['UB'][i] - graph['LB'][i]) <= 0.001 or i > 10:
            run = False
        i += 1

    # Plot UB and LB
    plt.plot(graph['UB'].keys(), graph['UB'].values(), label='Upper Bound')
    plt.plot(graph['LB'].keys(), graph['LB'].values(), label='Lower Bound')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


# Solve function
def Solve(model):
    opt = SolverFactory("gurobi")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions=True)
    return results, model


if __name__ == '__main__':
    main()
