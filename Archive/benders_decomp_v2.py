# Imports
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

"""
Benders decomposition for one scenario
"""

def main():
    file_name = '../Datasett_NO1_Cleaned_r5.xlsx'
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

"""Mathematical formulation 1st stage"""
def Obj_1st(m):
    return m.nuclear_DA * m.MC['nuclear'] + m.hydro_DA * m.MC['hydro'] + m.hydro_res_DA * m.C_res + m.alpha
def DA_load_balance(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand
def nuclear_lim(m):
    return m.P_min['nuclear'], m.nuclear_DA, m.P_max['nuclear']
def hydro_lim(m):
    return m.hydro_DA + m.hydro_res_DA <= m.P_max['hydro']

def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] - m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])

"""Master problem model formulation"""
def masterModel(data, Cuts):
    m = pyo.ConcreteModel()
    """Sets"""
    m.G             = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    """Parameters"""
    m.P_max         = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min         = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC            = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand        = pyo.Param(initialize=data['Consumers']['consumption']['Load 1'])
    m.wind_DA       = pyo.Param(initialize=45.6)
    m.C_res         = pyo.Param(initialize=25)
    """Variables"""
    m.nuclear_DA    = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_DA      = pyo.Var(within=pyo.NonNegativeReals)
    m.hydro_res_DA  = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, m.P_max['hydro']))
    """Cuts"""
    m.Cut           = pyo.Set(initialize=Cuts["Set"])  # Set for cuts
    m.Phi           = pyo.Param(m.Cut, initialize=Cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda        = pyo.Param(m.Cut, initialize=Cuts["lambda"])  # Parameter for lambda (dual value of reserve)
    m.X_hat         = pyo.Param(m.Cut, initialize=Cuts["x_hat"])  # Parameter for reserved hydro
    m.alpha         = pyo.Var(bounds= (-10000, 10000))  # Variable for alpha
    """Constraints"""
    m.DA_balance    = pyo.Constraint(rule=DA_load_balance)
    m.nuclear_lim   = pyo.Constraint( rule=nuclear_lim)
    m.hydro_lim     = pyo.Constraint(rule=hydro_lim)
    m.CreateCuts    = pyo.Constraint(m.Cut, rule=CreateCuts)
    """Objective Function"""
    m.obj           = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)
    return m

"""Mathematical formulation 2nd stage"""
def Obj_2nd(m):
    return m.hydro_RT * m.MC['hydro'] + m.rationing * m.C_rat
def RT_load_balance(m):
    return m.nuclear_RT + m.wind + m.hydro_RT + m.rationing >= m.demand
def hydro_RT_upper(m):
    return m.hydro_RT <= m.hydro_DA + m.X_hat
def rationing_limit(m):
    return 0, m.rationing, m.demand
# def hydro_RT_lower(m):
#     return m.hydro_RT >= m.hydro_DA - m.X_hat


"""Sub-problem model formulation"""
def subModel(data, X_hat, DA_values, wind):
    m = pyo.ConcreteModel()
    """Sets"""
    m.G             = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    # m.S             = pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')
    """Parameters"""
    m.MC            = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand        = pyo.Param(initialize=data['Consumers']['consumption']["Load 1"])
    m.C_rat         = pyo.Param(initialize=data['Consumers']['rationing_cost']["Load 1"])
    m.wind          = pyo.Param(initialize=wind)
    # m.prob          = pyo.Param(m.S, initialize={'low': 1 / 3, 'med': 1 / 3, 'high': 1 / 3})
    m.nuclear_RT    = pyo.Param(initialize=DA_values["nuclear_DA"])
    m.hydro_DA      = pyo.Param(initialize=DA_values["hydro_DA"])
    m.X_hat         = pyo.Param(initialize=X_hat)
    """Variables"""
    m.hydro_RT      = pyo.Var(within=pyo.NonNegativeReals)
    m.rationing     = pyo.Var(within=pyo.NonNegativeReals)
    """Constraints"""
    m.RT_balance    = pyo.Constraint(rule=RT_load_balance)
    m.hydro_RT_upper= pyo.Constraint(rule=hydro_RT_upper)
    m.rationing_lim = pyo.Constraint(rule=rationing_limit)
    # m.hydro_RT_lower= pyo.Constraint(rule=hydro_RT_lower)
    """Objective Function"""
    m.obj           = pyo.Objective(rule=Obj_2nd, sense=pyo.minimize)
    return m


def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m

def DisplayResults(m):
    return print(m.display(), m.dual.display())

def print_benders(m_1st, m_2nd):
    print("1st stage:")
    print(f"Master Objective: {pyo.value(m_1st.obj):.2f}")
    print("2nd stage:")
    print(f"Sub Objective: {pyo.value(m_2nd.obj):.2f}")
    print(f"Total: {pyo.value(m_1st.obj):.2f} - {pyo.value(m_2nd.obj):.2f} = {(pyo.value(m_1st.obj) - pyo.value(m_2nd.obj)):.2f}")



def manageCuts(Cuts, m):
    """Add new cut to existing dictionary of cut information"""
    cut = len(Cuts["Set"])
    Cuts['Set'].append(cut)
    Cuts['Phi'][cut]    = pyo.value(m.obj)
    Cuts['lambda'][cut] = m.dual[m.RT_balance]
    Cuts['x_hat'][cut]  = pyo.value(m.X_hat)
    # Cuts['Phi'][cut] = data_cuts["Phi"]
    # Cuts['lambda'][cut] = data_cuts["lambda"]
    # Cuts['x_hat'][cut] = data_cuts["x_hat"]
    return Cuts

def benders(data):
    """
    Setup for benders decomposition
    We perform this for x iterations
    """
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    graph = {}
    graph["UB"] = {}
    graph["LB"] = {}

    run = True
    i = 0
    while run:
        m_1st = masterModel(data, Cuts)
        Solve(m_1st)

        X_hat = m_1st.hydro_res_DA
        DA_values = {"nuclear_DA": pyo.value(m_1st.nuclear_DA),
                     "hydro_DA": pyo.value(m_1st.hydro_DA)
                     }

        wind = data["Time_wind"]["med"]

        print("Iteration", i)
        print(f"X_hat: {X_hat.value}")
        # data_cuts = {"Phi": 0, "lambda": 0, "x_hat": 0}
        # for s in data["Time_wind"].keys():
        #     m_2nd = subModel(data, X_hat, DA_values, data["Time_wind"][s])
        #     Solve(m_2nd)
        #     data_cuts["Phi"] += pyo.value(m_2nd.obj)/len(data["Time_wind"].keys())
        #     data_cuts["lambda"] += m_2nd.dual[m_2nd.reserve]/len(data["Time_wind"].keys())
        #     data_cuts["x_hat"] += pyo.value(m_2nd.X_hat)/len(data["Time_wind"].keys())
        m_2nd = subModel(data, X_hat, DA_values, wind)
        Solve(m_2nd)

        Cuts = manageCuts(Cuts, m_2nd)

        print("Objective function: ", pyo.value(m_2nd.obj))
        print("Cut information acquired:")
        for component in Cuts:
            print(component, Cuts[component])

        graph['UB'][i] = pyo.value(m_1st.alpha.value)
        graph['LB'][i] = pyo.value(m_2nd.obj)
        print("UB:", pyo.value(m_1st.alpha.value), "- LB:", pyo.value(m_2nd.obj))
        """Convergence check"""
        if (abs(pyo.value(m_1st.alpha.value) - pyo.value(m_2nd.obj)) <= 0.001) or i > 5:
            run = False
        i += 1
        input()

    print_benders(m_1st, m_2nd)
    DisplayResults(m_1st)
    DisplayResults(m_2nd)
    # Ploting the result
    plt.plot(graph['UB'].keys(), graph['UB'].values(), label='Upper Bound (UB)')
    plt.plot(graph['LB'].keys(), graph['LB'].values(), label='Lower Bound (LB)')
    plt.xlabel('Iterations')
    plt.ylabel('Euro')
    plt.title('UB and LB')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()