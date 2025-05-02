import re

def print_iteration(iteration):
    for line in iteration:
        if line.startswith("{"):
            data = json.loads(line)
            print(f"STEP: {data['step']:9,}, SIDE: {data['side']:6}, REWARD: {data['reward']: 6.4f}, DURATION: {data['trade_duration']:6,}")


def process_file(filename):
    line_count = 0
    current_iteration = []
    iterations = []
    with open(filename, 'r') as log:
        while line := log.readline():
            if bool(re.match(r"^\|\s*total_timesteps", line)):
                current_iteration.append(line)
                iterations.append(current_iteration)
                current_iteration = []
            else:
                current_iteration.append(line)
            line_count += 1
    return iterations

