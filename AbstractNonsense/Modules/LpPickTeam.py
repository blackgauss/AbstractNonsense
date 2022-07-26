import pulp
import pandas as pd
import numpy as np
from collections import Counter

def make_variables(df, name:str, criterion:str):
    return [ 
        {'var' : pulp.LpVariable(name + '_' + str(row_num), lowBound=0, upBound=1, cat='Integer'), \
         'Cost' : row['Cost'], \
         'xP' : row[criterion], \
         'Team' : row['Team'], \
         'Position' : row['Position'], \
         'Name' : row['Name'], 
         'ID' : row['ID']} \
         for row_num, row in df.iterrows() ]


def pick_team(xPdf, criterion : str, formation:tuple[int], total_cost:float):
    """Finds optimal FPL team from projected expected points.
    
    xPdf -- data frame with cost, xP, team, position, ID, and name information
    criterion -- columns which optimizer uses for xP players
    formation -- football formation used for starting 11
    total_cost -- maximum cost of squad
    
    As of now xPdf must use standardized position labels.
    Goalkeeper - 'GK'
    Defender - 'DEF'
    Midfielder - 'MID'
    Forward - 'FWD'
    """
    starter_formation_dict = {'GK' : 1, 'DEF' : formation[0], 'MID' : formation[1], 'FWD' : formation[2]}
    sub_formation_dict = {'GK' : 1, 'DEF' : 5 - formation[0], 'MID' : 5 - formation[1], 'FWD' : 3 - formation[2]}
    
    # LpProblem Set-Up
    score_modifiers = {'Starters' : 1, 'Bench' : 0, 'Captain' : 1} # How do selections contribute to GW score
    obj_func = 0 # Objective function to maximize score
    
    cost_func = 0
    
    squad_decisions = {'Starters' : [], 'Bench' : [], 'Captain' : []} # decision variables for LpProblem
    squad_totals = {'Starters' : 0, 'Bench' : 0, 'Captain' : 0}
    
    team_funcs = {}
    
    for team in xPdf['Team'].unique():
        team_funcs[team] = 0

    position_funcs = {'Starters' : {'GK' : 0, 'DEF' : 0, 'MID' : 0, 'FWD' : 0}, \
                      'Bench' : {'GK' : 0, 'DEF' : 0, 'MID' : 0, 'FWD' : 0}}
    
    for decision in squad_decisions:
        squad_decisions[decision] = make_variables(xPdf, decision, criterion)
        
        for variable_dict in squad_decisions[decision]:
            variable = variable_dict['var']
            xp = variable_dict['xP']
            cost = variable_dict['Cost']
            
            obj_func += score_modifiers[decision] * variable * xp
            squad_totals[decision] += variable

            # COST TEAM AND POSITION CONSTRAINT FUNCTIONS
            if decision in ['Starters', 'Bench']:
                cost_func += variable * cost
                
                team = variable_dict['Team']
                team_funcs[team] += variable
                
                position = variable_dict['Position']
                position_funcs[decision][position] += variable

    problem = pulp.LpProblem('Optimal_Team', pulp.LpMaximize)
    # CONSTRAINTS
    problem += obj_func, 'Objective'
    problem += (cost_func <= total_cost), 'Cost'
    
    
    problem += (squad_totals['Starters'] == 11), 'Starters'
    problem += (squad_totals['Bench'] == 4), 'Bench'
    problem += (squad_totals['Captain'] == 0), 'Captain'
    for team in team_funcs:
        problem += (team_funcs[team] <= 3), team

    for category in position_funcs:
        for position in position_funcs[category]:
            if category == 'Starters':
                problem += (position_funcs['Starters'][position] == starter_formation_dict[position]), 'starter_' + position
            if category == 'Bench':
                problem += (position_funcs['Bench'][position] == sub_formation_dict[position]), 'bench_' + position

    problem.solve()
    
    # RETURN DF
    starters = squad_decisions['Starters']
    benchs = squad_decisions['Bench']
    
    starter_df = []
    bench_df = []
    
    for starter in starters:
        if pulp.value(starter['var']) == 1.0:
            starter_df.append(starter)
    for bench in benchs:
        if pulp.value(bench['var']) == 1.0:
            bench_df.append(bench)
                    
    starters = pd.DataFrame(starter_df)[['Name', 'ID', 'Team', 'Position', 'Cost', 'xP']]
    bench = pd.DataFrame(bench_df)[['Name', 'ID', 'Team', 'Position', 'Cost', 'xP']]
    return_df = pd.concat([starters, bench], keys=['Starters', 'Bench'])
    
    return (squad_decisions, return_df, pulp.value(problem.objective))


def position_sort(solution):
    """In-house sorting function for optimal team solution.
    
    solution -- output from AbstractNonsense.pick_team()
    """
    starters = solution[0]['Starters']
    benchs = solution[0]['Bench']
    positions = ['GK', 'DEF', 'MID', 'FWD']
    squad_dict = {'GK' : {'Starters' : [], 'Bench' : []}, \
                  'DEF' : {'Starters' : [], 'Bench' : []}, \
                  'MID' : {'Starters' : [], 'Bench' : []}, \
                  'FWD' : {'Starters' : [], 'Bench' : []}}
    
    for position in positions:
        for starter in starters:
            if pulp.value(starter['var']) == 1.0:
                if starter['position'] == position:
                    squad_dict[position]['Starters'].append([starter['Name'], starter['Team'], starter['Cost'], starter['xP']])
        for bench in benchs:
            if pulp.value(bench['var']) == 1.0:
                if bench['position'] == position:
                    squad_dict[position]['Bench'].append([bench['Name'], bench['Team'], bench['Cost'], bench['xP']])
    
    return squad_dict