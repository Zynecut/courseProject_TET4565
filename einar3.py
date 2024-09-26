from pyomo.environ import *
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory

# Definer manuell innmelt produksjon for første dagen
manual_nuclear_day1 = 200  # Manuell produksjon for nuclear dag 1
manual_hydro_day1 = 30     # Manuell produksjon for hydro dag 1

# Vindscenarioer med produksjon for dag 1 og dag 2
wind_scenario_values = {
    ('wind_med', '02/06/2020'): 20.32,
    ('wind_med', '03/06/2020'): 45.6,
    ('wind_high', '02/06/2020'): 20.32,
    ('wind_high', '03/06/2020'): 55.904,
    ('wind_low', '02/06/2020'): 20.32,
    ('wind_low', '03/06/2020'): 34.504
}

# Opprette modellen
model = ConcreteModel()

# Set - Strukturerer indekser
model.Generators = Set(initialize=["nuclear", "hydro", "wind"])
model.Loads = Set(initialize=["Load 1"])
model.Scenarios = Set(initialize=["wind_med", "wind_high", "wind_low"])
model.Time = Set(initialize=["02/06/2020", "03/06/2020"])

# Parametere - Definerer konstanter
model.P_max = Param(model.Generators, initialize={"nuclear": 200, "hydro": 60, "wind": 80})
model.P_min = Param(model.Generators, initialize={"nuclear": 0, "hydro": 0, "wind": 0})
model.MC = Param(model.Generators, initialize={"nuclear": 15, "hydro": 30, "wind": 0})
model.P_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 20, "wind": 0})
model.C_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 30, "wind": 0})
model.Load_Demand = Param(model.Loads, initialize={"Load 1": 250})
model.C_rationing = Param(initialize=2500)
model.P_wind = Param(model.Scenarios, model.Time, initialize=wind_scenario_values)

# Variabler
model.Production = Var(model.Generators, model.Time, model.Scenarios, within=NonNegativeReals)
model.Rationing = Var(model.Loads, model.Time, model.Scenarios, within=NonNegativeReals)
model.Hydro_DA = Var(model.Time, within=NonNegativeReals)
model.Nuclear_DA = Var(model.Time, within=NonNegativeReals)

# Begrensning for produksjonen
def prod_lim_max(model, g, t, s):
    return model.Production[g, t, s] <= model.P_max[g]
model.ProdMaxCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_max)

def prod_lim_min(model, g, t, s):
    return model.Production[g, t, s] >= model.P_min[g]
model.ProdMinCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_min)

def wind_prod_limit(model, t, s):
    return model.Production['wind', t, s] <= model.P_wind[s, t]
model.WindProdLimit = Constraint(model.Time, model.Scenarios, rule=wind_prod_limit)

# Frys dag 1 produksjon basert på manuell input
def freeze_day1_nuclear(model, s):
    return model.Production["nuclear", "02/06/2020", s] == manual_nuclear_day1
model.FreezeDay1Nuclear = Constraint(model.Scenarios, rule=freeze_day1_nuclear)

def freeze_day2_nuclear(model, s):
    return model.Production["nuclear", "03/06/2020", s] == manual_nuclear_day1
model.FreezeDay2Nuclear = Constraint(model.Scenarios, rule=freeze_day2_nuclear)

def freeze_day1_hydro(model, s):
    return model.Production["hydro", "02/06/2020", s] == manual_hydro_day1
model.FreezeDay1Hydro = Constraint(model.Scenarios, rule=freeze_day1_hydro)


# Real-time justering av hydro for dag 2 (nedre og øvre grense)
def hydro_real_time_adjustment_lower(model, t, s):
    if t == "02/06/2020":
        return Constraint.Skip  # Ingen justering for dag 1
    return model.Production["hydro", t, s] >= manual_hydro_day1 - model.P_reserved["hydro"]  # Minimum hydro på dag 2
model.HydroRealTimeAdjustmentLowerCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_lower)

def hydro_real_time_adjustment_upper(model, t, s):
    if t == "02/06/2020":
        return Constraint.Skip  # Ingen justering for dag 1
    return model.Production["hydro", t, s] <= manual_hydro_day1 + model.P_reserved["hydro"]  # Maksimum hydro på dag 2
model.HydroRealTimeAdjustmentUpperCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_upper)

# Lastbalansering
def load_balance_cons(model, l, t, s):
    return sum(model.Production[g, t, s] for g in model.Generators) + model.Rationing[l, t, s] >= model.Load_Demand[l]
model.LoadBalanceCons = Constraint(model.Loads, model.Time, model.Scenarios, rule=load_balance_cons)

# Objektivfunksjon
def objective_rule(model):
    production_cost = sum(model.Production[g, t, s] * model.MC[g] for g in model.Generators for t in model.Time for s in model.Scenarios)
    reserved_cost = sum(model.P_reserved["hydro"] * model.C_reserved["hydro"] for t in model.Time)
    rationing_cost = sum(model.Rationing[l, t, s] * model.C_rationing for l in model.Loads for t in model.Time for s in model.Scenarios)
    return production_cost + reserved_cost + rationing_cost
model.Objective = Objective(rule=objective_rule, sense=minimize)

# Løsning
model.dual = Suffix(direction=Suffix.IMPORT)
solver = SolverFactory('glpk')
solver.solve(model, tee=True)

# Hente ut resultater
results = []
for t in model.Time:
    for s in model.Scenarios:
        nuclear_prod = model.Production['nuclear', t, s].value
        hydro_prod = model.Production['hydro', t, s].value
        wind_prod = model.Production['wind', t, s].value
        rationing_value = model.Rationing['Load 1', t, s].value
        reserved_hydro = model.P_reserved['hydro']
        results.append({
            'Time': t,
            'Scenario': s,
            'Nuclear Production (MW)': nuclear_prod,
            'Hydro Production (MW)': hydro_prod,
            'Wind Production (MW)': wind_prod,
            'Rationing (MW)': rationing_value,
            'Hydro Reserved for Next Day (MW)': reserved_hydro
        })

# Presentere resultater
df_results = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(df_results)
