# Benders decomposition for one scenario
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt


def main():
    file_name = 'Datasett_NO1_Cleaned_r5.xlsx'
    data = inputData(file_name)
    benders(data)


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


def DA_load_balance(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand


def nuclear_lim(m):
    return m.P_min['nuclear'], m.nuclear_DA, m.P_max['nuclear']


def hydro_lim(m):
    return m.P_min['hydro'], m.hydro_DA, m.P_max['hydro']


def hydro_res_min(m):
    return m.hydro_res_DA >= 0


def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] - m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])


def masterModel(data, Cuts):
    m = pyo.ConcreteModel()

    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')

    """Parameters"""
    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(initialize=data['Consumers']['consumption']['Load 1'])
    m.wind_DA = pyo.Param(initialize=45.6)  # Hardcode for now
    m.C_res = pyo.Param(initialize=25)

    """Variables"""
    m.nuclear_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.alpha = pyo.Var(bounds=(-100000, 100000))  # For cuts

    """Cuts"""
    m.Cut = pyo.Set(initialize=Cuts["Set"])  # Set for cuts
    m.Phi = pyo.Param(m.Cut, initialize=Cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda = pyo.Param(m.Cut, initialize=Cuts["lambda"])  # Parameter for lambda (dual value of reserve)
    m.X_hat = pyo.Param(m.Cut, initialize=Cuts["x_hat"])  # Parameter for reserved hydro

    """Constraints"""
    m.DA_balance = pyo.Constraint(rule=DA_load_balance)
    m.nuclear_lim = pyo.Constraint(rule=nuclear_lim)
    m.hydro_lim = pyo.Constraint(rule=hydro_lim)
    m.HydroResMin = pyo.Constraint(rule=hydro_res_min)
    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)

    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)
    return m


"""Sub-problem formulation"""


def Obj_2nd(m):
    return m.hydro_RT * m.MC['hydro'] + m.rationing * m.C_rat


def RT_load_balance(m):
    return m.nuclear_RT + m.wind + m.hydro_DA + m.hydro_RT + m.rationing >= m.demand


def hydro_RT_limit(m):
    return -m.X_hat, m.hydro_RT, m.X_hat


def rationing_limit(m):
    return 0, m.rationing, m.demand


# ** New hydro constraints from deterministic model **
def hydro_upper_RT(m):
    """Upper bound for hydro_RT: hydro_RT <= hydro_DA + hydro_res_DA"""
    return m.hydro_RT <= m.hydro_DA + m.hydro_res_DA


def hydro_lower_RT(m):
    """Lower bound for hydro_RT: hydro_RT >= hydro_DA - hydro_res_DA"""
    return m.hydro_RT >= m.hydro_DA - m.hydro_res_DA


def subModel(data, X_hat, DA_values, wind):
    m = pyo.ConcreteModel()

    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')

    """Parameters"""
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(initialize=data['Consumers']['consumption']["Load 1"])
    m.C_rat = pyo.Param(initialize=data['Consumers']['rationing_cost']["Load 1"])
    m.wind = pyo.Param(initialize=wind)
    m.nuclear_RT = pyo.Param(initialize=DA_values["nuclear_DA"])
    m.hydro_DA = pyo.Param(initialize=DA_values["hydro_DA"])
    m.hydro_res_DA = pyo.Param(initialize=DA_values["hydro_res_DA"])
    m.X_hat = pyo.Param(initialize=X_hat)

    """Variables"""
    m.hydro_RT = pyo.Var(within=pyo.NonNegativeReals)
    m.rationing = pyo.Var(within=pyo.NonNegativeReals)

    """Constraints"""
    m.RT_balance = pyo.Constraint(rule=RT_load_balance)

    # Handling the case where X_hat is zero (no reserve)
    if pyo.value(X_hat) == 0:
        m.hydro_RT_eq = pyo.Constraint(expr=m.hydro_RT == m.hydro_DA)
    else:
        m.hydro_upper_RT = pyo.Constraint(rule=hydro_upper_RT)
        m.hydro_lower_RT = pyo.Constraint(rule=hydro_lower_RT)

    m.rationing_lim = pyo.Constraint(rule=rationing_limit)

    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_2nd, sense=pyo.minimize)

    return m


def manageCuts(Cuts, m):
    """Add new cut to existing dictionary of cut information"""
    cut = len(Cuts["Set"])
    Cuts['Set'].append(cut)
    Cuts['Phi'][cut] = pyo.value(m.obj)
    Cuts['lambda'][cut] = m.dual[m.RT_balance]  # Endret fra m.reserve til m.hydro_upper_RT
    Cuts['x_hat'][cut] = pyo.value(m.X_hat)
    return Cuts


def display_results_benders(m_1st, m_2nd):
    """Display the results for the current iteration"""
    print("\n--- Master Problem Results ---")
    print(f"Nuclear DA: {pyo.value(m_1st.nuclear_DA)}")
    print(f"Hydro DA: {pyo.value(m_1st.hydro_DA)}")
    print(f"Hydro Reserve DA: {pyo.value(m_1st.hydro_res_DA)}")
    print(f"Objective Function (Master): {pyo.value(m_1st.obj)}")

    print("\n--- Subproblem Results ---")
    print(f"Hydro RT: {pyo.value(m_2nd.hydro_RT)}")
    print(f"Rationing: {pyo.value(m_2nd.rationing)}")
    print(f"Objective Function (Subproblem): {pyo.value(m_2nd.obj)}")


def benders(data):
    """
    Setup for benders decomposition
    """
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    graph = {}
    graph["UB"] = {}
    graph["LB"] = {}

    for i in range(10):
        print(f"\n--- Iteration {i + 1} ---")
        m_1st = masterModel(data, Cuts)
        Solve(m_1st)

        X_hat = pyo.value(m_1st.hydro_res_DA)
        DA_values = {"nuclear_DA": pyo.value(m_1st.nuclear_DA), "hydro_DA": pyo.value(m_1st.hydro_DA),
                     "hydro_res_DA": pyo.value(m_1st.hydro_res_DA)}
        wind = data['Time_wind']['low']

        print(f"X_hat (Hydro Reserve DA): {X_hat}")

        m_2nd = subModel(data, X_hat, DA_values, wind)
        results, m_2nd = Solve(m_2nd)

        # Check if the subproblem was solved successfully
        if results.solver.termination_condition == 'infeasible':
            print("Subproblem infeasible. Skipping iteration.")
            break

        Cuts = manageCuts(Cuts, m_2nd)

        # Store upper and lower bounds for plotting
        graph['UB'][i] = pyo.value(m_1st.alpha)
        graph['LB'][i] = pyo.value(m_2nd.obj)

        # Display the results of this iteration
        display_results_benders(m_1st, m_2nd)

        """Convergence check"""
        if abs(graph['UB'][i] - graph['LB'][i]) <= 0.001:
            break

    # Plotting the result
    plt.plot(graph['UB'].keys(), graph['UB'].values(), label='Upper Bound (UB)')
    plt.plot(graph['LB'].keys(), graph['LB'].values(), label='Lower Bound (LB)')
    plt.xlabel('Iterations')
    plt.ylabel('Euro')
    plt.title('UB and LB')
    plt.legend()
    plt.show()


def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m


if __name__ == '__main__':
    main()
