import csv
import numpy
from sklearn import preprocessing

import pulp

seasons = [
    "2021-22",
    # "2020-21"
]

teams = {}
players = []
player_codes = []
ignore_player_codes = []

positions = {
    1: {
        "type": "GKP",
        "squad_select": 2,
    },
    2: {
        "type": "DEF",
        "squad_select": 5,
    },
    3: {
        "type": "MID",
        "squad_select": 5,
    },
    4: {
        "type": "FWD",
        "squad_select": 3,
    }
}

data_names = []
data_cost = []
data_total_points = []
data_cost_effective = []
data_time_effective = []
data_team_strength = []
data_my_score = []
data_players = {}
data_is_gkp = []
data_is_def = []
data_is_mid = []
data_is_fwd = []
picked_codes = []


def data_cleanup():
    data_names.clear()
    data_cost.clear()
    data_total_points.clear()
    data_cost_effective.clear()
    data_time_effective.clear()
    data_team_strength.clear()
    data_my_score.clear()
    data_players.clear()
    data_is_gkp.clear()
    data_is_def.clear()
    data_is_mid.clear()
    data_is_fwd.clear()


def create_teams_dict():
    for season in seasons:
        with open("data/" + season + "/teams.csv", newline='') as csvfile:
            teams_reader = csv.DictReader(csvfile)
            for row in teams_reader:
                teams[row['code']] = {
                    'name': row['name'],
                    'short_name': row['short_name'],
                    'points': int(row['points']),
                    'position': int(row['position']),
                    'strength_overall': int(row['strength_overall_home']) + int(row['strength_overall_away']),
                    'strength_overall_home': int(row['strength_overall_home']),
                    'strength_overall_away': int(row['strength_overall_away']),
                    'strength_attack': int(row['strength_attack_home']) + int(row['strength_attack_away']),
                    'strength_attack_home': int(row['strength_attack_home']),
                    'strength_attack_away': int(row['strength_attack_away']),
                    'strength_defence': int(row['strength_defence_home']) + int(row['strength_defence_away']),
                    'strength_defence_home': int(row['strength_defence_home']),
                    'strength_defence_away': int(row['strength_defence_away'])
                }


def create_players_dict():
    for season in seasons:
        with open("data/" + season + "/players_raw.csv", newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['code'] not in player_codes and int(row['code']) not in ignore_player_codes:
                    players.append({
                        'first_name': row['first_name'],
                        'second_name': row['second_name'],
                        'web_name': row['web_name'],
                        'code': int(row['code']),
                        'total_points': int(row['total_points']),
                        'minutes': int(row['minutes']),
                        'now_cost': int(row['now_cost']),
                        'element_type': positions[int(row['element_type'])]['type'],
                        'bps': int(row['bps']),
                        'bonus': int(row['bonus']),
                        'team_short': teams[row['team_code']]['short_name'],
                        'team': teams[row['team_code']]['name'],
                        'team_strength': teams[row['team_code']]['strength_overall'],
                        'cost_effective': int(row['total_points']) / int(row['now_cost']),
                        'time_effective': calculate_time_effectiveness(row),
                        'my_score': int(row['total_points']) / int(row['now_cost']) + calculate_time_effectiveness(row)
                    })
                player_codes.append(row['code'])


def calculate_time_effectiveness(row):
    minutes = int(row['minutes'])
    if minutes == 0:
        return 0
    return int(row['total_points']) / int(row['minutes'])


def set_position(index, position):
    data_is_gkp.insert(index, 0)
    data_is_def.insert(index, 0)
    data_is_mid.insert(index, 0)
    data_is_fwd.insert(index, 0)
    if position == 'GKP':
        data_is_gkp.insert(index, 1)
    elif position == 'DEF':
        data_is_def.insert(index, 1)
    elif position == 'MID':
        data_is_mid.insert(index, 1)
    elif position == 'FWD':
        data_is_fwd.insert(index, 1)


def normalize_data(player_codes_to_ignore):
    data_cleanup()
    for i, player in enumerate(players):
        if player['code'] not in player_codes_to_ignore:
            key = player['web_name'] + '_' + player["team_short"]
            key = key.replace("-", "_")
            key = key.replace(" ", "_")

            data_names.insert(i, key)
            data_cost.insert(i, player['now_cost'])
            data_total_points.insert(i, player['total_points'])
            data_cost_effective.insert(i, player['cost_effective'])
            data_team_strength.insert(i, player['team_strength'])
            set_position(i, player['element_type'])
            data_players[key] = {
                "Player": player['web_name'],
                "Team": player['team'],
                "Position": player['element_type'],
                "Cost": player['now_cost'],
                "Total": player['total_points'],
                "Code": player['code']
            }
    data_cost_effective_normalized = preprocessing.normalize(numpy.array(data_cost_effective).reshape(1, -1))
    data_team_strength_normalized = preprocessing.normalize(numpy.array(data_team_strength).reshape(1, -1))
    data_my_score.extend(numpy.add(data_cost_effective_normalized, data_team_strength_normalized).flatten().tolist())
    print(len(data_my_score))
    print("Ignored {} players".format(len(player_codes_to_ignore)))
    print("Normalized {} players".format(len(data_names)))


iteration_config = {
    0: {
        "total_cost": 1000,
        "gkp": 1,
        "def": 3,
        "mid": 4,
        "fwd": 3,
        "maximize": "data_total_points"
    },
    # 1: {
    #     "total_cost": 260,
    #     "gkp": 1,
    #     "def": 1,
    #     "mid": 3,
    #     "fwd": 0,
    #     "maximize": "data_cost_effective"
    # }
}


def calculate(player_codes_to_ignore, i):
    normalize_data(player_codes_to_ignore)
    print("MyScore: " + str(data_my_score))
    LpVariableList = [pulp.LpVariable('{}'.format(item), lowBound=0, upBound=1) for item in data_names]

    problem = pulp.LpProblem(name="Total Points Maximizer", sense=pulp.LpMaximize)
    problem += pulp.lpDot(data_my_score, LpVariableList)
    problem += pulp.lpDot(data_cost, LpVariableList) <= iteration_config[i]['total_cost']
    problem += pulp.lpDot(data_is_gkp, LpVariableList) <= iteration_config[i]['gkp']
    problem += pulp.lpDot(data_is_def, LpVariableList) <= iteration_config[i]['def']
    problem += pulp.lpDot(data_is_mid, LpVariableList) <= iteration_config[i]['mid']
    problem += pulp.lpDot(data_is_fwd, LpVariableList) <= iteration_config[i]['fwd']

    # Solve
    print("Solving for config: " + str(iteration_config[i]))
    status = problem.solve()
    print(pulp.LpStatus[status])

    # Result
    print("Result:")
    total_cost = 0
    total_points = 0
    picked_players = 0
    picked_per_position = {
        "GKP": [],
        "DEF": [],
        "MID": [],
        "FWD": []
    }
    for player in LpVariableList:
        if int(player.value()) > 0:
            p = data_players[str(player)]
            total_cost += p['Cost']
            total_points += p['Total']
            picked_codes.append(p['Code'])
            picked_players += 1
            picked_per_position[p['Position']].append(p)

    print("Total cost: " + str(total_cost))
    print("Total points: " + str(total_points))
    print("Picked players: " + str(picked_players))
    print("Picked GKP: " + str(len(picked_per_position['GKP'])))
    print(*picked_per_position['GKP'], sep="\n")
    print("Picked DEF: " + str(len(picked_per_position['DEF'])))
    print(*picked_per_position['DEF'], sep="\n")
    print("Picked MID: " + str(len(picked_per_position['MID'])))
    print(*picked_per_position['MID'], sep="\n")
    print("Picked FWD: " + str(len(picked_per_position['FWD'])))
    print(*picked_per_position['FWD'], sep="\n")


def main():
    create_teams_dict()
    create_players_dict()
    print("Gathered {} teams".format(len(teams)))
    print("Gathered {} players".format(len(players)))
    calculate([], 0)
    # calculate(picked_codes, 1)


if __name__ == '__main__':
    main()
