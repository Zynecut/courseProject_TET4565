# Imports
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

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

def Obj_1st(m):
    production_cost_DA = m.hydro_res_DA * const['MC_res'] + m.hydro_DA * m.MC['hydro'] + m.nuclear_DA * m.MC['nuclear']
    return production_cost_DA + m.alpha

"""Set Variable Bounds"""
def limit_nuclear_DA(m):
    return m.P_min["nuclear"], m.P_max["nuclear"]

def limit_hydro_DA(m):
    return m.P_min["hydro"], m.P_max["hydro"]

def load_balance_DA(m):
    return m.nuclear_DA + m.hydro_DA + m.wind_DA == m.demand["Load 1"]

def hydro_res_min(m):
    return m.hydro_res_DA >= 0

def CreateCuts(m, c):
    return m.alpha >= m.Phi[c] + m.Lambda[c] * (m.hydro_res_DA - m.X_hat[c])

# def CreateCuts(m, c):
#     # Inkluder både produksjon (hydro_DA) og reservoar (hydro_res_DA) i kuttene
#     return m.alpha >= m.Phi[c] + m.Lambda_DA[c] * (m.hydro_DA - m.X_hat_DA[c]) + m.Lambda_res[c] * (m.hydro_res_DA - m.X_hat_res[c])

def Model_1st(data, Cuts):
    m = pyo.ConcreteModel()

    """Sets"""
    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')

    """Parameters"""
    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.wind_DA = pyo.Param(initialize=data['Time_wind']['med'])

    """Variables"""
    m.nuclear_DA = pyo.Var(bounds=limit_nuclear_DA, within=pyo.NonNegativeReals)
    m.hydro_DA = pyo.Var(bounds=limit_hydro_DA, within=pyo.NonNegativeReals)
    m.hydro_res_DA = pyo.Var(within=pyo.NonNegativeReals)
    m.wind_prod_DA = pyo.Var(within=pyo.NonNegativeReals)

    """Cuts"""
    m.Cut = pyo.Set(initialize=Cuts["Set"])  # Set for cuts
    m.Phi = pyo.Param(m.Cut, initialize=Cuts["Phi"])  # Parameter for Phi (Objective cost)
    m.Lambda = pyo.Param(m.Cut, initialize=Cuts["lambda"])  # Parameter for lambda (dual value of reservoir)
    m.X_hat = pyo.Param(m.Cut, initialize=Cuts["x_hat"])  # Parameter for reservoir level

    m.alpha = pyo.Var(bounds=(-100000, 100000))

    """Constraints"""
    m.LoadBalance_DA = pyo.Constraint(rule=load_balance_DA)
    m.HydroResMin = pyo.Constraint(rule=hydro_res_min)
    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)

    """Objective Function"""
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.minimize)

    return m

def limit_nuclear_RT(m, s):
    return m.P_min["nuclear"], m.P_max["nuclear"]

def limit_hydro_RT(m, s):
    return m.P_min["hydro"], m.P_max["hydro"]

def rationing_limits(m, l, s):
    return 0, m.demand[l]

def load_balance_RT(m, s):
    return m.hydro_RT[s] + m.wind_prod_RT[s] + m.nuclear_RT[s] + m.rationing["Load 1", s] >= m.demand["Load 1"]

def hydro_upper_RT(m, s):
    return m.hydro_RT[s] <= m.hydro_DA + m.hydro_res_DA

def hydro_lower_RT(m, s):
    return m.hydro_RT[s] >= m.hydro_DA - m.hydro_res_DA

def ObjFunction(m):
    production_cost_RT = sum(m.prob[s] * (m.MC['hydro'] * (m.hydro_RT[s]-m.hydro_DA)) for s in m.S)
    rationing_cost = sum(m.prob[s] * m.cost_rat[l] * m.rationing[l, s] for l in m.L for s in m.S)
    return production_cost_RT + rationing_cost

def Model_2nd(data, X_hat, hydro_DA_val, hydro_res_DA_val):
    m = pyo.ConcreteModel()

    m.G = pyo.Set(initialize=list(data['Producers']['p_max'].keys()))  # ('nuclear', 'hydro', 'wind')
    m.L = pyo.Set(initialize=list(data['Consumers']['consumption'].keys()))  # ('Load 1')
    m.S = pyo.Set(initialize=list(data['Time_wind'].keys()))  # ('low', 'med', 'high')

    m.P_max = pyo.Param(m.G, initialize=data['Producers']['p_max'])
    m.P_min = pyo.Param(m.G, initialize=data['Producers']['p_min'])
    m.MC = pyo.Param(m.G, initialize=data['Producers']['marginal_cost'])
    m.demand = pyo.Param(m.L, initialize=data['Consumers']['consumption'])
    m.cost_rat = pyo.Param(m.L, initialize=data['Consumers']['rationing_cost'])
    m.P_wind = pyo.Param(m.S, initialize=data['Time_wind'])
    m.prob = pyo.Param(m.S, initialize={'low': 1 / 3, 'med': 1 / 3, 'high': 1 / 3})
    m.X_hat = pyo.Param(initialize=X_hat)

    m.hydro_DA = pyo.Param(initialize=hydro_DA_val)
    m.hydro_res_DA = pyo.Param(initialize=hydro_res_DA_val)

    m.nuclear_RT = pyo.Var(m.S, bounds=limit_nuclear_RT, within=pyo.NonNegativeReals)
    m.hydro_RT = pyo.Var(m.S, bounds=limit_hydro_RT, within=pyo.NonNegativeReals)
    m.wind_prod_RT = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.rationing = pyo.Var(m.L, m.S, bounds=rationing_limits, within=pyo.NonNegativeReals)

    m.LoadBalance_RT = pyo.Constraint(m.S, rule=load_balance_RT)
    m.HydroUpper_RT = pyo.Constraint(m.S, rule=hydro_upper_RT)
    m.HydroLower_RT = pyo.Constraint(m.S, rule=hydro_lower_RT)

    m.WindProdRTConstraint = pyo.Constraint(m.S, rule=lambda m, s: m.wind_prod_RT[s] == m.P_wind[s])

    # """Definer dual suffix"""
    # m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.obj = pyo.Objective(rule=ObjFunction, sense=pyo.minimize)

    return m

# Solve function
def SolveModel(model):
    opt = SolverFactory("gurobi")
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions=True)
    return results, model

def benders(data):
    # Initielle kutt og parametere
    Cuts = {"Set": [], "Phi": {}, "lambda": {}, "x_hat": {}}

    upper_bounds = []  # Liste for å lagre øvre grenser
    lower_bounds = []  # Liste for å lagre nedre grenser

    epsilon = 1e-5  # Toleranse for konvergenssjekk
    prev_obj_value = float('inf')  # Tidligere objektivfunksjonsverdi for sammenligning

    # Iterasjonsloop
    for iteration in range(1, 100):
        print(f"Iteration: {iteration}")

        # Første-stegs modell (master problem)
        model_1st = Model_1st(data, Cuts)
        results, model_1st = SolveModel(model_1st)

        # Lower Bound: objektiv funksjonsverdi fra første stegs modell (inkluderer alpha som kuttene tilfører)
        lower_bound = pyo.value(model_1st.obj)
        lower_bounds.append(lower_bound)

        # Lagre X_hat (hydro_res_DA verdi) fra første stegs løsning
        X_hat = pyo.value(model_1st.hydro_res_DA)
        hydro_DA_val = pyo.value(model_1st.hydro_DA)
        hydro_res_DA_val = pyo.value(model_1st.hydro_res_DA)

        # Andre-stegs modell (subproblem)
        model_2nd = Model_2nd(data, X_hat, hydro_DA_val, hydro_res_DA_val)
        results, model_2nd = SolveModel(model_2nd)

        # Upper Bound: Kombinasjon av første-stegs objektivverdi og andre-stegs løsning
        upper_bound = lower_bound + pyo.value(model_2nd.obj)
        upper_bounds.append(upper_bound)

        # Hent gjennomsnittlig dualverdi for LoadBalance_RT constraint for alle scenarier
        dual_lambda = sum(model_2nd.dual[model_2nd.LoadBalance_RT[s]] for s in model_2nd.S) / len(model_2nd.S)

        # Generer kutt basert på dualvariabler
        # Oppdater kutt-settet med verdier
        Cuts["Set"].append(iteration)
        Cuts["Phi"][iteration] = pyo.value(model_2nd.obj)  # Phi er objektiv funksjonsverdien fra subproblemet
        Cuts["lambda"][iteration] = dual_lambda
        Cuts["x_hat"][iteration] = X_hat

        # Konvergenssjekk: Hvis endringen i objektivfunksjonen er mindre enn epsilon, avslutt
        if abs(prev_obj_value - pyo.value(model_2nd.obj)) < epsilon:
            print("Convergence achieved.")
            break

        prev_obj_value = pyo.value(model_2nd.obj)

    # Print endelige resultater
    print("\nFinal Results:")
    print(f"Final Phi (objective value): {pyo.value(model_2nd.obj)}")
    print(f"Final Lambda (dual value): {dual_lambda}")
    print(f"Final Hydro_res_DA (X_hat): {X_hat}")

    # Plot upper and lower bounds
    plot_bounds(upper_bounds, lower_bounds)

def plot_bounds(upper_bounds, lower_bounds):
    iterations = list(range(len(upper_bounds)))

    plt.plot(iterations, upper_bounds, label="Upper Bound (UB)", color="blue")
    plt.plot(iterations, lower_bounds, label="Lower Bound (LB)", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Euro")
    plt.title("UB and LB")
    plt.legend()
    plt.grid(True)
    plt.show()

# Display results
def DisplayModelResults(model):
    """
    Funksjonen viser modellens variabler og dualverdier hvis tilgjengelig.
    """
    # Viser modellens variabler og deres verdier
    model.display()

    # Hvis dualverdier er definert, vis dem også
    if hasattr(model, 'dual'):
        print("\nDual values:")
        for constraint in model.component_objects(pyo.Constraint, active=True):
            print(f"Dual for {constraint}:")
            for index in constraint:
                print(f"  {index}: {model.dual[constraint[index]]}")


if __name__ == '__main__':
    main()
