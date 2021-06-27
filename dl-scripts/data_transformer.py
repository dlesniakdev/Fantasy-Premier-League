import pulp
import csv

season = "2021-22"
dataPath = "data/" + season + "/"

teams = {}
players = []
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
data_players = {}

def create_teams_dict():
    with open(dataPath + "teams.csv", newline='') as csvfile:
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
    with open(dataPath + "players_raw.csv", newline='') as csvfile:
        player_reader = csv.DictReader(csvfile)
        for row in player_reader:
            players.append({
                'first_name': row['first_name'],
                'second_name': row['second_name'],
                'web_name': row['web_name'],
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
                'time_effective': calculate_time_effectiveness(row)
            })


def calculate_time_effectiveness(row):
    minutes = int(row['minutes'])
    if minutes == 0:
        return 0
    return int(row['total_points']) / int(row['minutes'])


def normalize_data():
    for i, player in enumerate(players):
        key = player['web_name'] + '_' + player["team_short"]
        key = key.replace("-", "_")
        key = key.replace(" ", "_")

        data_names.insert(i, key)
        data_cost.insert(i, player['now_cost'])
        data_total_points.insert(i, player['total_points'])
        data_players[key] = {"Player": player['web_name'], "Team": player['team'], "Position": player['element_type'], "Cost": player['now_cost'], "Total": player['total_points']}


def calculate():
    normalize_data()

    LpVariableList = [pulp.LpVariable('{}'.format(item), lowBound=0, upBound=1) for item in data_names]

    problem = pulp.LpProblem(name="Total Points", sense=pulp.LpMaximize)
    problem += pulp.lpDot(data_total_points, LpVariableList)
    problem += pulp.lpDot(data_cost, LpVariableList) <= 1000
    problem += pulp.lpDot([1 for i in data_names], LpVariableList) <= 15

    print(pulp.lpDot(data_cost, LpVariableList))
    print(pulp.lpDot([1 for i in data_names], LpVariableList))
    pass
    # Solve
    status = problem.solve()
    print(pulp.LpStatus[status])

    # Result
    print("Result:")
    total_cost = 0
    total_points = 0
    picked_players = 0
    for player in LpVariableList:
        if int(player.value()) > 0:
            p = data_players[str(player)]
            total_cost += p['Cost']
            total_points += p['Total']
            picked_players += 1
            print(str(player) + " × " + str(int(player.value())) + " | " + str(p))
    print("Total cost: " + str(total_cost))
    print("Total points: " + str(total_points))
    print("Picked players: " + str(picked_players))

def main():
    create_teams_dict()
    create_players_dict()
    calculate()


if __name__ == '__main__':
    main()
