from pyomo.environ import *
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory

# Vindscenarioer med produksjon for real-time
wind_scenario_values = {
    'wind_low': 34.504,
    'wind_med': 45.6,
    'wind_high': 55.904
}

# Opprette modellen
model = ConcreteModel()

# Set - Strukturerer indekser
model.Generators = Set(initialize=["nuclear", "hydro", "wind"])
model.Loads = Set(initialize=["Load 1"])
model.Scenarios = Set(initialize=["wind_low", "wind_med", "wind_high"])

# Parametere - Definerer konstanter
model.P_max = Param(model.Generators, initialize={"nuclear": 200, "hydro": 60, "wind": 80})
model.P_min = Param(model.Generators, initialize={"nuclear": 0, "hydro": 0, "wind": 0})
model.MC = Param(model.Generators, initialize={"nuclear": 15, "hydro": 30, "wind": 0})
model.P_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 20, "wind": 0})
model.C_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 30, "wind": 0})
model.Load_Demand = Param(model.Loads, initialize={"Load 1": 250})
model.Prob = Param(model.Scenarios, initialize={"wind_low": 0.6, "wind_med": 0.3, "wind_high": 0.1})
model.C_rationing = Param(initialize=250)
model.P_wind = Param(model.Scenarios, initialize=wind_scenario_values)

# Variabler for produksjon i day-ahead og real-time
model.Production_DA = Var(model.Generators, within=NonNegativeReals)  # Day-Ahead produksjon
model.Production_RT = Var(model.Generators, within=NonNegativeReals)  # Real-Time produksjon
model.Rationing_RT = Var(model.Loads, within=NonNegativeReals)  # Rasjonering i real-time
model.Wind_Spilled = Var(within=NonNegativeReals)  # Tapt vindproduksjon i real-time

# Real-time vindparameter (initielt ukjent, oppdateres etter stage 1)
model.P_wind_RT = Param(initialize=0, mutable=True)

# Beregn forventet vindproduksjon i day-ahead
def expected_wind_production(model):
    return sum(model.P_wind[s] * model.Prob[s] for s in model.Scenarios)

# Begrensninger for stage 1 (day-ahead)
def wind_projection(model):
    # Setter forventet vindproduksjon i day-ahead til å være lik den beregnede forventningsverdien
    return model.Production_DA['wind'] == expected_wind_production(model)
model.WindProjection = Constraint(rule=wind_projection)

# Begrensninger for maks og min produksjon for day-ahead
def prod_lim_max_da(model, g):
    return model.Production_DA[g] <= model.P_max[g]
model.ProdMaxConsDA = Constraint(model.Generators, rule=prod_lim_max_da)

def prod_lim_min_da(model, g):
    return model.Production_DA[g] >= model.P_min[g]
model.ProdMinConsDA = Constraint(model.Generators, rule=prod_lim_min_da)

# Load balance for day-ahead
def load_balance_day_ahead(model):
    return sum(model.Production_DA[g] for g in model.Generators) >= model.Load_Demand["Load 1"]
model.LoadBalanceDayAhead = Constraint(rule=load_balance_day_ahead)

# Objective function for day-ahead beslutninger
def day_ahead_objective_rule(model):
    # Første stage kostnad (day-ahead beslutninger)
    day_ahead_cost = sum(model.Production_DA[g] * model.MC[g] for g in model.Generators)
    return day_ahead_cost
model.DayAheadObjective = Objective(rule=day_ahead_objective_rule, sense=minimize)

# Begrensninger for hydro-produksjon i real-time basert på fleksibilitet
def hydro_real_time_adjustment_lower(model):
    return model.Production_RT["hydro"] >= model.Production_DA["hydro"] - model.P_reserved["hydro"]  # Hydro-produksjon kan justeres ned
model.HydroRealTimeAdjustmentLowerCons = Constraint(rule=hydro_real_time_adjustment_lower)

def hydro_real_time_adjustment_upper(model):
    return model.Production_RT["hydro"] <= model.Production_DA["hydro"] + model.P_reserved["hydro"]  # Hydro-produksjon kan justeres opp
model.HydroRealTimeAdjustmentUpperCons = Constraint(rule=hydro_real_time_adjustment_upper)

# --- Første stage (Day-Ahead) ---
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# --- Første stage resultater (Day-Ahead) ---
print("\n--- Stage 1: Day-Ahead Results ---")
for g in model.Generators:
    print(f"{g.capitalize()} Day-Ahead Production: {model.Production_DA[g].value:.4f} MW")

# --- Stage 2 (Real-Time) ---
# Her legger du inn den faktiske vindproduksjonen fra den kommende dagen.
realized_wind_prod = float(input("\nEnter realized wind production for the day (MW): "))

# Oppdater real-time vindparameteren med den faktiske verdien
model.P_wind_RT.set_value(realized_wind_prod)

# Begrensninger for real-time
def wind_prod_limit_real_time(model):
    return model.Production_RT['wind'] + model.Wind_Spilled == model.P_wind_RT  # Vindprod + tapt vind = vind tilgjengelig
model.WindProdLimitRT = Constraint(rule=wind_prod_limit_real_time)

# Begrensninger for maks og min produksjon for real-time
def prod_lim_max_rt(model, g):
    return model.Production_RT[g] <= model.P_max[g]
model.ProdMaxConsRT = Constraint(model.Generators, rule=prod_lim_max_rt)

def prod_lim_min_rt(model, g):
    return model.Production_RT[g] >= model.P_min[g]
model.ProdMinConsRT = Constraint(model.Generators, rule=prod_lim_min_rt)

# Load balance for real-time
def load_balance_real_time(model):
    return sum(model.Production_RT[g] for g in model.Generators) + model.Rationing_RT['Load 1'] >= model.Load_Demand['Load 1']
model.LoadBalanceRealTime = Constraint(rule=load_balance_real_time)

# Real-time objective function
def real_time_objective_rule(model):
    # Andre stage kostnad (real-time beslutninger)
    real_time_cost = sum(
        model.Production_RT[g] * model.MC[g] for g in model.Generators
    ) + model.Rationing_RT['Load 1'] * model.C_rationing
    return real_time_cost
model.RealTimeObjective = Objective(rule=real_time_objective_rule, sense=minimize)

# --- Overgang mellom stage 1 og stage 2 ---
# Deaktiver day-ahead målsetning
model.DayAheadObjective.deactivate()

# Aktiver real-time målsetning
model.RealTimeObjective.activate()

# Løs real-time fasen
results = solver.solve(model, tee=True)

# --- Andre stage resultater (Real-Time) ---
print("\n--- Stage 2: Real-Time Results ---")
for g in model.Generators:
    print(f"{g.capitalize()} Real-Time Production: {model.Production_RT[g].value:.4f} MW")
wind_spilled = model.Wind_Spilled.value
rationing_value = model.Rationing_RT['Load 1'].value
print(f"Wind Spilled: {wind_spilled:.4f} MW")
print(f"Rationing: {rationing_value:.4f} MW")
