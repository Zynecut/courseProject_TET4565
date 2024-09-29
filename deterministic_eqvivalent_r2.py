from pyomo.environ import *
from pyomo.opt import SolverFactory

# Definer modellen
model = ConcreteModel()

# Sett med scenarier
model.producers = Set(initialize=["nuclear", "hydro", "wind"])
model.consumers = Set(initialize=["Load 1"])
model.scenarios = Set(initialize = ['low', 'medium', 'high'])

# Parametere
model.P_max = Param(model.producers, initialize={"nuclear": 200, "hydro": 60, "wind": 80})
model.P_min = Param(model.producers, initialize={"nuclear": 0, "hydro": 0, "wind": 0})
model.MC = Param(model.producers, initialize={"nuclear": 15, "hydro": 30, "wind": 0})
model.P_wind = Param(model.scenarios, initialize={'low': 34.504, 'medium': 45.6, 'high': 55.904})
model.probabilities = Param(model.scenarios, initialize= {'low': 0, 'medium': 0, 'high': 1})
model.demand = Param(model.consumers, initialize={"Load 1": 250})
model.cost_rat = Param(model.consumers, initialize={"Load 1": 250})                                             # Kostnad ved rasjonering


# Beregner forventet vindproduksjon
def expected_wind_production(model):
    return sum(model.P_wind[s] * model.probabilities[s] for s in model.scenarios)                                       # Beregner forventet vindproduksjon basert på sannsynlighetene for de ulike scenarioene.
expected_wind_production(model)


# Variabler
def production_limits(model, p):
    return (model.P_min[p], model.P_max[p])
model.Production = Var(model.producers, bounds=production_limits, within=NonNegativeReals)                      # Setter øvre og nedre begrensning for produksjon direkte i variabelen ved initialisering.

def rationing_limits(model, l):
    return (0, model.demand[l])
model.Rationing = Var(model.consumers, bounds=rationing_limits, within=NonNegativeReals)                        # Setter øvre og nedre begrensning for rasjonering direkte i variabelen ved initialisering.


# Constraints
def load_balance(model):
    return sum(model.Production[p] for p in model.producers) + sum(model.Rationing[l] for l in model.consumers) == model.demand["Load 1"]
model.LoadBalance = Constraint(rule=load_balance)

def wind_limitation(model):
    return model.Production['wind'] <= expected_wind_production(model)                                                  # Begrenser vindproduksjon til kalkulert forventning.
model.WindLimitation = Constraint( rule=wind_limitation)



# Objective function
def OBJ(model):
    production_cost = sum(model.MC[p] * model.Production[p] for p in model.producers)
    rationig_cost = sum(model.cost_rat[l] * model.Rationing[l] for l in model.consumers)
    return production_cost + rationig_cost
model.objective = Objective(rule=OBJ, sense=minimize)


opt = SolverFactory('glpk')
result = opt.solve(model)

print(model.display())