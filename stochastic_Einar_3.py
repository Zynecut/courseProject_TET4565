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
model.MC = Param(model.producers, initialize={"nuclear": 15, "hydro": 30, "wind": 5})
model.P_wind = Param(model.scenarios, initialize={'low': 34.504, 'medium': 45.6, 'high': 55.904})
model.probabilities = Param(model.scenarios, initialize= {'low': 0.8, 'medium': 0.1, 'high': 0.1})
model.demand = Param(model.consumers, initialize={"Load 1": 250})
model.cost_rat = Param(model.consumers, initialize={"Load 1": 2500})                                             # Kostnad ved rasjonering



# Begrenser produksjon samtidige som jeg oppretter variabeler
def limit_nuclear_DA(model):
    return (model.P_min["nuclear"], model.P_max["nuclear"])                                                             # Begrenser produksjonen til maksimal produksjon for kjernekraft.
model.nuclear_DA = Var(bounds=limit_nuclear_DA, within=NonNegativeReals)

def limit_hydro_DA(model):
    return (model.P_min["hydro"], model.P_max["hydro"])
model.hydro_DA = Var(bounds=limit_hydro_DA, within=NonNegativeReals)

def limit_wind_DA(model):
    return (model.P_min["wind"], model.P_max["wind"])
model.wind_DA = Var(bounds=limit_wind_DA, within=NonNegativeReals)

def limit_nuclear_RT(model, s):
    return (model.P_min["nuclear"], model.P_max["nuclear"])
model.nuclear_RT = Var(model.scenarios, bounds=limit_nuclear_RT, within=NonNegativeReals)

def limit_hydro_RT(model, s):
    return (model.P_min["hydro"], model.P_max["hydro"])
model.hydro_RT = Var(model.scenarios, bounds=limit_hydro_RT, within=NonNegativeReals)

def limit_wind_RT(model, s):
    return (model.P_min["wind"], model.P_max["wind"])
model.wind_RT = Var(model.scenarios, bounds=limit_wind_RT, within=NonNegativeReals)



model.wind_surplus = Var(model.scenarios, within=NonNegativeReals)

def wind_surplus_constraint(model, s):
    return model.wind_surplus[s] >= model.wind_DA - model.P_wind[s]
model.WindSurplusConstraint = Constraint(model.scenarios, rule=wind_surplus_constraint)





def locked_nuclear_prod(model, s):
    return model.nuclear_DA == model.nuclear_RT[s]
model.LockedNuclearProd = Constraint(model.scenarios, rule=locked_nuclear_prod)



# RASJONERING
def rationing_limits(model, l, s):
    return (0, model.demand[l])
model.rationing = Var(model.consumers, model.scenarios, bounds=rationing_limits, within=NonNegativeReals)                        # Setter øvre og nedre begrensning for rasjonering direkte i variabelen ved initialisering.


def load_balance_DA(model):
    return model.nuclear_DA + model.hydro_DA + model.wind_DA == model.demand["Load 1"]
model.LoadBalance_DA = Constraint(rule=load_balance_DA)

def load_balance_RT(model, s):
    return model.hydro_RT[s] + model.wind_RT[s] + model.rationing["Load 1", s] == model.demand["Load 1"]
model.LoadBalance_RT = Constraint(model.scenarios, rule=load_balance_RT)





def OBJ(model):
    production_cost_DA = model.MC['nuclear'] * model.nuclear_DA + model.MC['hydro'] * model.hydro_DA + model.MC['wind'] * model.wind_DA
    production_cost_RT = sum(model.probabilities[s] * (model.MC['nuclear'] * model.nuclear_RT[s] + model.MC['hydro'] * model.hydro_RT[s] + model.MC['wind'] * model.wind_RT[s]) for s in model.scenarios)
    rationing_cost = sum(model.probabilities[s] * model.cost_rat[l] * model.rationing[l, s] for l in model.consumers for s in model.scenarios)
    risk_cost = sum(model.probabilities[s] * 50 * model.wind_surplus[s] for s in model.scenarios)                       # Prøvde å legge inn en straffekostnad for overskuddsvind
    return production_cost_DA + production_cost_RT + rationing_cost + risk_cost
model.objective = Objective(rule=OBJ, sense=minimize)


opt = SolverFactory('glpk')
result = opt.solve(model, tee=True)

print(model.display())