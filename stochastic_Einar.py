from pyomo.environ import *
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory


# 20.32
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



# Variabler
model.Production = Var(model.Generators, model.Time, model.Scenarios, within=NonNegativeReals)
model.Rationing = Var(model.Loads, model.Time, model.Scenarios, within=NonNegativeReals)
model.Nuclear_DA = Var(model.Time, within=NonNegativeReals)                                                     # Beslutning som ikke er avhengig av hva som skjer real_time, men som er basert på vekting av forventede scenarioer.
model.Hydro_DA = Var(model.Time, within=NonNegativeReals)                                                       # Beslutning som ikke er avhengig av hva som skjer real_time, men som er basert på vekting av forventede scenarioer.
model.Wind_DA = Var(model.Time, within=NonNegativeReals)                                                        # Beslutning som ikke er avhengig av hva som skjer real_time, men som er basert på vekting av forventede scenarioer.



# Constraints
def prod_lim_max(model, g, t, s):
    return model.Production[g, t, s] <= model.P_max[g]
model.ProdMaxCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_max)

def prod_lim_min(model, g, t, s):
    return model.Production[g, t, s] >= model.P_min[g]
model.ProdMinCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_min)

def wind_prod_limit(model, s):
    return model.Production['wind', 'real_time', s] <= model.P_wind[s]                                                         # Sørger for at vindproduksjonen ikke overstiger tilgjengelig vind i hvert scenario.
model.WindProdLimit = Constraint( model.Scenarios, rule=wind_prod_limit)

def locked_nuclear_prod(model, s):
    return model.Production["nuclear", "day_ahead", s] == model.Production["nuclear", "real_time", s]
model.LockedNuclearProd = Constraint(model.Scenarios, rule=locked_nuclear_prod)

def hydro_real_time_adjustment_lower(model, t, s):
    if t == "day_ahead":
        return Constraint.Skip  # Ingen justering for dag 1
    else:
        return model.Production["hydro", "day_ahead", s] - model.P_reserved["hydro"] <= model.Production["hydro", t, s]        # Bruker produksjonen fra dagen før som et utgangspunkt og setter en nedre grense for nedjustering.
model.HydroRealTimeAdjustmentLowerCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_lower)

def hydro_real_time_adjustment_upper(model, t, s):
    if t == "day_ahead":
        return Constraint.Skip  # Ingen justering for dag 1
    else:
        return model.Production["hydro", "day_ahead", s] + model.P_reserved["hydro"] >= model.Production["hydro", t, s]         # Bruker produksjonen fra dagen før som et utgangspunkt og setter en øvre grense for oppjustering.
model.HydroRealTimeAdjustmentUpperCons = Constraint(model.Time, model.Scenarios, rule=hydro_real_time_adjustment_upper)




# Load balance
def load_balance_cons(model, l, t, s):
    return sum(model.Production[g, t, s] for g in model.Generators) + model.Rationing[l, t, s] >= model.Load_Demand[l]
model.LoadBalanceCons = Constraint(model.Loads, model.Time, model.Scenarios, rule=load_balance_cons)


#  MÅ FÅ INN PROBABILITY!!

# Objective
def objective_rule(model):
    production_cost = sum(model.Production[g, t, s] * model.MC[g] * model.Prob[s] for g in model.Generators for t in model.Time for s in model.Scenarios)
    reserved_cost = sum(model.P_reserved["hydro"] * model.C_reserved["hydro"] for t in model.Time)
    rationing_cost = sum(model.Rationing[l, t, s] * model.C_rationing * model.Prob[s] for l in model.Loads for t in model.Time for s in model.Scenarios)
    return production_cost + reserved_cost + rationing_cost
model.Objective = Objective(rule=objective_rule, sense=minimize)


# Solve
model.dual = Suffix(direction=Suffix.IMPORT)
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Print full model structure
print("\n--- Model Structure ---")
model.pprint()

# Print dual values (only constraints with a dual value)
print("\n--- Dual Values ---")
for c in model.component_objects(Constraint, active=True):
    print("\nConstraint:", c)
    for index in c:
        if c[index].active and index in model.dual:
            print(f"  {index} : Dual = {model.dual[c[index]]}")

# Print variable values
print("\n--- Variable Values ---")
for v in model.component_objects(Var, active=True):
    print(f"\nVariable: {v}")
    varobject = getattr(model, str(v))
    for index in varobject:
        print(f"  {index} : Value = {varobject[index].value}")





# # Get results in a DataFrame for printing
# results = []
# for t in model.Time:
#     for s in model.Scenarios:
#         nuclear_prod = model.Production['nuclear', t, s].value
#         hydro_prod = model.Production['hydro', t, s].value
#         wind_prod = model.Production['wind', t, s].value
#         rationing_value = model.Rationing['Load 1', t, s].value
#         reserved_hydro = model.P_reserved['hydro']
#         wind_spilled = model.P_wind[s] - wind_prod  # Juster for at P_wind ikke har tidsindeks
#         results.append({
#             'Time': t,
#             'Scenario': s,
#             'Nuclear Production (MW)': nuclear_prod,
#             'Hydro Production (MW)': hydro_prod,
#             'Wind Production (MW)': wind_prod,
#             'Wind Spilled (MW)': wind_spilled,
#             'Rationing (MW)': rationing_value,
#             'Hydro Reserved for Next Day (MW)': reserved_hydro
#         })
#
# # Juster Pandas visningsinnstillinger for å vise hele DataFrame
# pd.set_option('display.max_columns', None)  # Vis alle kolonner
# pd.set_option('display.expand_frame_repr', False)  # Ikke avkort kolonner
#
# # Present results
# df_results = pd.DataFrame(results)
# print(df_results)



#
# # Konsolider resultatene for stage_1
# day1_results = df_results[df_results['Time'] == 'stage_1']
#
# # Sjekk om alle verdier for stage_1 er like
# if day1_results.iloc[:, 2:].nunique().max() == 1:
#     # Hvis alle verdier er like, behold kun én rad
#     df_results_day1 = day1_results.drop_duplicates(subset=['Time'], keep='first')
#     # Endre scenario-navnet til 'wind_today' ved å bruke .loc
#     df_results_day1.loc[:, 'Scenario'] = 'Day-ahead prediction'
# else:
#     # Hvis ikke, behold alle rader for dag 1
#     df_results_day1 = day1_results
#
# # Kombiner stage_1-resultater med stage_2-resultater
# df_results_day2 = df_results[df_results['Time'] == 'stage_2']
# df_results_combined = pd.concat([df_results_day1, df_results_day2])
#
# # Print combined results for both stages
# print("Combined Results for Stage 1 and Stage 2:")
# print(df_results_combined)

# Now calculate and print the cost for each producer per scenario for both stages

# # Stage 1 costs (stage_1)
# print("\nProducer Costs for Stage 1 (stage_1):")
# for s in model.Scenarios:
#     producer_costs_stage1 = {g: model.Production[g, "stage_1", s].value * model.MC[g] for g in model.Generators}
#
#     # Create a DataFrame for producer costs for stage_1
#     df_producer_costs_stage1 = pd.DataFrame({
#         'Producer': list(producer_costs_stage1.keys()),
#         'Production Cost (Euro)': list(producer_costs_stage1.values())
#     })
#
#     # Print producer costs for stage 1
#     print(f"\nProducer Costs for stage_1 - Scenario: {s}")
#     print(df_producer_costs_stage1)
#
# # Stage 2 costs (stage_2)
# print("\nProducer Costs for Stage 2 (stage_2):")
# for s in model.Scenarios:
#     producer_costs_stage2 = {g: model.Production[g, "stage_2", s].value * model.MC[g] for g in model.Generators}
#
#     # Create a DataFrame for producer costs for stage_2
#     df_producer_costs_stage2 = pd.DataFrame({
#         'Producer': list(producer_costs_stage2.keys()),
#         'Production Cost (Euro)': list(producer_costs_stage2.values())
#     })
#
#     # Print producer costs for stage 2
#     print(f"\nProducer Costs for stage_2 - Scenario: {s}")
#     print(df_producer_costs_stage2)
#
