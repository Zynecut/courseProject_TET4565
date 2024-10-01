
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Function to load data using the provided inputData method
def inputData(file):
    # read data from file
    data = {}
    excel_sheets = ['Producers', 'Consumers', 'Time_wind']
    for sheet in excel_sheets:
        df = pd.read_excel(file, sheet_name=sheet)
        data[sheet] = df.to_dict(orient='list')
    return data

# Load the data and check the keys in 'Time_wind'
data = inputData('Datasett_NO1_Cleaned_r5.xlsx')

#Defining probability
data['Prob'] = {'Low': 1/3, 'Avg': 1/3, 'High': 1/3}

const = {'res_cap' : 20, 'MC_res' : 30, 'MC_rat' : 250, 'MC_w' : 0, 'MC_nuc' : 30, 'MC_h' : 30}
const_2 = {'MAX_nuc' : 200, 'MAX_h' : 60, 'MAX_w' : 80, 'MIN' : 0}

def Obj(m):
    return const['MC_res'] * m_reserve['02.06.2020'] + sum(m.prob[s]*(m.windprod['03.06.2020',s]*const['MC_w'] + m.rationing['03.06.2020',s]*const['MC_rat']) for s in m.s)








