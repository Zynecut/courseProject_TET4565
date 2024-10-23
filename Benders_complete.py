# Benders decomposition for multiple scenarios
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
    m.hydro_DA      = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_res_DA  = pyo.Var(within=pyo.NonNegativeReals)
    m.alpha         = pyo.Var(bounds=(-10000, 10000))
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
    return sum(m.prob[s] * (m.hydro_RT[s] * m.MC['hydro'] + m.rationing[s] * m.C_rat) for s in m.S)
"""Constraints"""
def RT_load_balance(m, s):
    return m.hydro_RT[s] + m.wind_RT[s] + m.nuclear_RT[s] + m.rationing[s] >= m.demand
def hydro_link_RT(m, s):
    """link hydro_RT with hydro_DA ± hydro_res_DA"""
    return m.hydro_RT[s] <= m.hydro_DA + m.X_hat


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
    m.hydro_link_RT = pyo.Constraint(m.S, rule=hydro_link_RT)
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


def display_results_benders(m_1st, m_2nd):
    """Display the results for the current iteration"""
    print("\n--- Master Problem Results ---")
    print(f"Nuclear DA: {pyo.value(m_1st.nuclear_DA):.2f}")
    print(f"Hydro DA: {pyo.value(m_1st.hydro_DA):.2f}")
    print(f"Hydro Reserve DA: {pyo.value(m_1st.hydro_res_DA):.2f}")
    print(f"Objective Function (Master-Problem): {pyo.value(m_1st.obj):.2f}")
    print("\n--- Sub-Problem Results ---")
    for s in m_2nd.S:
        print(f"Scenario {s} - Wind: {pyo.value(m_2nd.wind_RT[s]):.2f}")  # Print wind for each scenario
        print(f"Scenario {s} - Hydro RT: {pyo.value(m_2nd.hydro_RT[s]):.2f}")
        print(f"Scenario {s} - Rationing: {pyo.value(m_2nd.rationing[s]):.2f}")
    print(f"Objective Function (Sub-Problem): {pyo.value(m_2nd.obj):.2f}")
    print("\n--- Total Objective Value ---")
    print(f"{pyo.value(m_1st.obj):.2f} - {pyo.value(m_2nd.obj):.2f} = {(pyo.value(m_1st.obj) - pyo.value(m_2nd.obj)):.2f}")


def benders(data):
    """Setup for benders decomposition"""
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    graph = {}
    graph["UB"] = {}
    graph["LB"] = {}

    for i in range(10):
        m_1st = masterModel(data, Cuts)
        Solve(m_1st)

        X_hat = pyo.value(m_1st.hydro_res_DA)
        DA_values = {"nuclear_DA": pyo.value(m_1st.nuclear_DA), "hydro_DA": pyo.value(m_1st.hydro_DA)}

        probability = {'low': 0.3, 'med': 0.4, 'high': 0.3}
        m_2nd = subModel(data, X_hat, DA_values, probability)
        results, m_2nd = Solve(m_2nd)

        # Check if the subproblem was solved successfully
        if results.solver.termination_condition == 'infeasible':
            print("Subproblem infeasible. Skipping iteration.")
            break

        Cuts = manageCuts(Cuts, m_2nd)

        # Store upper and lower bounds for plotting
        graph['LB'][i] = pyo.value(m_1st.alpha)
        graph['UB'][i] = pyo.value(m_2nd.obj)

        # Display the results of this iteration
        print(f"\n------ Iteration {i + 1} ------")
        print(f"X_hat (Hydro Reserve DA): {X_hat:.2f}")
        # for component in Cuts:
        #     print(component, Cuts[component])
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