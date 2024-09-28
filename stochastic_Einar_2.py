from pyomo.environ import *
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory


# Vindscenarioer med produksjon for dag 1 og dag 2
# wind_scenario_values = {
#     ('wind_med', 'stage_1'): 20.32,
#     ('wind_med', 'stage_2'): 45.6,
#     ('wind_high', 'stage_1'): 20.32,
#     ('wind_high', 'stage_2'): 55.904,
#     ('wind_low', 'stage_1'): 20.32,
#     ('wind_low', 'stage_2'): 34.504
# }

wind_scenario_values = {
    ('wind_low'): 34.504,
    ('wind_med'): 45.6,
    ('wind_high'): 55.904
}

# Opprette modellen
model = ConcreteModel()

# Set - Strukturerer indekser
model.Generators = Set(initialize=["nuclear", "hydro", "wind"])
model.Loads = Set(initialize=["Load 1"])
model.Scenarios = Set(initialize=["wind_low", "wind_med", "wind_high"])
model.Time = Set(initialize=["day_ahead", "real_time"])

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

model.P_wind_RT = Param( model.Time, initialize=45)

# Variabler
model.Hydro_DA = Var(model.Time, within=NonNegativeReals)
model.Nuclear_DA = Var(model.Time, within=NonNegativeReals)
model.Wind_DA = Var(model.Time, within=NonNegativeReals)

model.Production = Var(model.Generators, model.Time, within=NonNegativeReals)
model.Rationing = Var(model.Loads, model.Time, model.Scenarios, within=NonNegativeReals)


def expected_wind_production(model):
    return sum(model.P_wind[s] * model.Prob[s] for s in model.Scenarios)                                                # Beregner forventet vindproduksjon (39.9728
expected_wind_production(model)


# Constraints
def prod_lim_max(model, g, t):
    return model.Production[g, t] <= model.P_max[g]
model.ProdMaxCons = Constraint(model.Generators, model.Time, rule=prod_lim_max)

def prod_lim_min(model, g, t):
    return model.Production[g, t] >= model.P_min[g]
model.ProdMinCons = Constraint(model.Generators, model.Time, rule=prod_lim_min)

def wind_prod_limit(model, t):
    return model.Production['wind', t] <= expected_wind_production(model)                                                         # Sørger for at vindproduksjonen ikke overstiger tilgjengelig vind i hvert scenario.
model.WindProdLimit = Constraint(model.Time, rule=wind_prod_limit)

def locked_nuclear_prod(model):
    return model.Production["nuclear", "day_ahead"] == model.Production["nuclear", "real_time"]
model.LockedNuclearProd = Constraint(rule=locked_nuclear_prod)

def hydro_real_time_adjustment_lower(model, t, s):
    if t == "02/06/2020":
        return Constraint.Skip  # Ingen justering for dag 1
    else:
        return model.Production["hydro", "02/06/2020", s] - model.P_reserved["hydro"] <= model.Production["hydro", t, s]        # Bruker produksjonen fra dagen før som et utgangspunkt og setter en nedre grense for nedjustering.
model.HydroRealTimeAdjustmentLowerCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_lower)

def hydro_real_time_adjustment_upper(model, t, s):
    if t == "02/06/2020":
        return Constraint.Skip  # Ingen justering for dag 1
    else:
        return model.Production["hydro", "02/06/2020", s] + model.P_reserved["hydro"] >= model.Production["hydro", t, s]         # Bruker produksjonen fra dagen før som et utgangspunkt og setter en øvre grense for oppjustering.
model.HydroRealTimeAdjustmentUpperCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_upper)