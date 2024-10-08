from pyomo.environ import *
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory


# 20.32
# Vindscenarioer med produksjon for dag 1 og dag 2
wind_scenario_values = {
    ('wind_med', '02/06/2020'): 45.6,
    ('wind_med', '03/06/2020'): 45.6,
    ('wind_high', '02/06/2020'): 55.904,
    ('wind_high', '03/06/2020'): 55.904,
    ('wind_low', '02/06/2020'): 34.504,
    ('wind_low', '03/06/2020'): 34.504
}

# Opprette modellen
model = ConcreteModel()

# Set - Strukturerer indekser
model.Generators = Set(initialize=["nuclear", "hydro", "wind"])
model.Loads = Set(initialize=["Load 1"])
model.Scenarios = Set(initialize=["wind_low", "wind_med", "wind_high"])
model.Time = Set(initialize=["02/06/2020", "03/06/2020"])

# Parametere - Definerer konstanter
model.P_max = Param(model.Generators, initialize={"nuclear": 200, "hydro": 60, "wind": 80})
model.P_min = Param(model.Generators, initialize={"nuclear": 0, "hydro": 0, "wind": 0})
model.MC = Param(model.Generators, initialize={"nuclear": 15, "hydro": 30, "wind": 0})
model.P_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 20, "wind": 0})
model.C_reserved = Param(model.Generators, initialize={"nuclear": 0, "hydro": 30, "wind": 0})
model.Load_Demand = Param(model.Loads, initialize={"Load 1": 250})
model.C_rationing = Param(initialize=250)
model.P_wind = Param(model.Scenarios, model.Time, initialize=wind_scenario_values)

# Variabler
model.Production = Var(model.Generators, model.Time, model.Scenarios, within=NonNegativeReals)
model.Rationing = Var(model.Loads, model.Time, model.Scenarios, within=NonNegativeReals)
model.Hydro_DA = Var(model.Time, within=NonNegativeReals)
model.Nuclear_DA = Var(model.Time, within=NonNegativeReals)



# Constraints
def prod_lim_max(model, g, t, s):
    return model.Production[g, t, s] <= model.P_max[g]
model.ProdMaxCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_max)

def prod_lim_min(model, g, t, s):
    return model.Production[g, t, s] >= model.P_min[g]
model.ProdMinCons = Constraint(model.Generators, model.Time, model.Scenarios, rule=prod_lim_min)

def wind_prod_limit(model, t, s):
    return model.Production['wind', t, s] <= model.P_wind[s, t]                                                         # Sørger for at vindproduksjonen ikke overstiger tilgjengelig vind i hvert scenario.
model.WindProdLimit = Constraint(model.Time, model.Scenarios, rule=wind_prod_limit)

def locked_nuclear_prod(model, s):
    return model.Production["nuclear", "02/06/2020", s] == model.Production["nuclear", "03/06/2020", s]
model.LockedNuclearProd = Constraint(model.Scenarios, rule=locked_nuclear_prod)


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



# Load balance
def load_balance_cons(model, l, t, s):
    return sum(model.Production[g, t, s] for g in model.Generators) + model.Rationing[l, t, s] >= model.Load_Demand[l]
model.LoadBalanceCons = Constraint(model.Loads, model.Time, model.Scenarios, rule=load_balance_cons)

# Objective
def objective_rule(model):
    production_cost = sum(model.Production[g, t, s] * model.MC[g] for g in model.Generators for t in model.Time for s in model.Scenarios)
    reserved_cost = sum(model.P_reserved["hydro"] * model.C_reserved["hydro"] for t in model.Time)
    rationing_cost = sum(model.Rationing[l, t, s] * model.C_rationing for l in model.Loads for t in model.Time for s in model.Scenarios)
    return production_cost + reserved_cost + rationing_cost
model.Objective = Objective(rule=objective_rule, sense=minimize)

# Solve
model.dual = Suffix(direction=Suffix.IMPORT)
solver = SolverFactory('glpk')
solver.solve(model, tee=True)

# Get results in a DataFrame for printing
results = []
for t in model.Time:
    for s in model.Scenarios:
        nuclear_prod = model.Production['nuclear', t, s].value
        hydro_prod = model.Production['hydro', t, s].value
        wind_prod = model.Production['wind', t, s].value
        rationing_value = model.Rationing['Load 1', t, s].value
        reserved_hydro = model.P_reserved['hydro']
        wind_spilled = model.P_wind[s, t] - wind_prod  # Beregn vind som ikke blir produsert
        results.append({
            'Time': t,
            'Scenario': s,
            'Nuclear Production (MW)': nuclear_prod,
            'Hydro Production (MW)': hydro_prod,
            'Wind Production (MW)': wind_prod,
            'Wind Spilled (MW)': wind_spilled,
            'Rationing (MW)': rationing_value,
            'Hydro Reserved for Next Day (MW)': reserved_hydro
        })

# Present results
df_results = pd.DataFrame(results)

# Juster Pandas visningsinnstillinger for å vise hele DataFrame
pd.set_option('display.max_columns', None)  # Vis alle kolonner
pd.set_option('display.expand_frame_repr', False)  # Ikke avkort kolonner

# Konsolider resultatene for 02/06/2020
day1_results = df_results[df_results['Time'] == '02/06/2020']

# Sjekk om alle verdier for 02/06/2020 er like
if day1_results.iloc[:, 2:].nunique().max() == 1:
    # Hvis alle verdier er like, behold kun én rad
    df_results_day1 = day1_results.drop_duplicates(subset=['Time'], keep='first')
    # Endre scenario-navnet til 'wind_today' ved å bruke .loc
    df_results_day1.loc[:, 'Scenario'] = 'Meldt inn day-ahead'
else:
    # Hvis ikke, behold alle rader for dag 1
    df_results_day1 = day1_results

# Kombiner dag 1-resultater med dag 2-resultater
df_results_day2 = df_results[df_results['Time'] == '03/06/2020']
df_results_combined = pd.concat([df_results_day1, df_results_day2])

# Print combined results
print(df_results_combined)

# Now calculate and print the cost for each producer per scenario for 03/06/2020
for s in model.Scenarios:
    producer_costs = {g: model.Production[g, "03/06/2020", s].value * model.MC[g] for g in model.Generators}

    # Create a DataFrame for producer costs for the specific scenario
    df_producer_costs = pd.DataFrame({
        'Producer': list(producer_costs.keys()),
        'Production Cost (Euro)': list(producer_costs.values())
    })

    # Print producer costs for the specific scenario
    print(f"\nProducer Costs for 03/06/2020 - Scenario: {s}")
    print(df_producer_costs)