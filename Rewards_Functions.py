import numpy as np
import click
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, timezone


def compute_reward_basic(action, current_price, previous_price, open_price, position_open, steps_in_trade=0, trades_today=0):
    # BASIC REWARDS
    reward = 0.0
    price_trend = current_price - previous_price

    if action == 0:  # HOLD
        if not position_open:
            # Penalize holding when no position is open and price is rising
            reward = -0.01 if price_trend > 0 else 0.01
        else:
            # Reward based on unrealized return
            reward = (current_price - open_price) / open_price

    elif action == 1:  # OPEN
        if not position_open:
            reward = 0.0  # No immediate reward for opening
        else:
            reward = -0.05  # Penalty for trying to open when already open

    elif action == 2:  # CLOSE
        if position_open:
            trade_return = (current_price - open_price) / open_price
            reward = trade_return

            # Add bonus/penalty depending on profit or loss
            if trade_return > 0:
                reward += 0.05  # Small bonus for profitable exit
            else:
                reward -= 0.05  # Small penalty for losing exit
        else:
            reward = -0.05  # Penalty for closing when no position is open

    return reward


def compute_reward_trade_freq(action, current_price, previous_price, open_price, position_open, steps_in_trade=0, trades_today=0):
    # ADJUSTED FOR TRADING FREQUENCY & HOLDING TIME
    reward = 0.0
    price_trend = current_price - previous_price

    # Parameters
    max_hold_penalty = -0.02
    max_trades_per_day = 5
    overtrade_penalty = -0.02
    hold_penalty_per_step = -0.001

    if action == 0:  # HOLD
        if not position_open:
            reward = -0.01 if price_trend > 0 else 0.01
        else:
            trade_return = (current_price - open_price) / open_price
            hold_time_penalty = min(steps_in_trade * hold_penalty_per_step, abs(max_hold_penalty))
            reward = trade_return + hold_time_penalty

    elif action == 1:  # OPEN
        if not position_open:
            reward = 0.0
            if trades_today >= max_trades_per_day:
                reward += overtrade_penalty  # discourage opening too often
        else:
            reward = -0.05  # trying to open when already open

    elif action == 2:  # CLOSE
        if position_open:
            trade_return = (current_price - open_price) / open_price
            reward = trade_return
            reward += 0.05 if trade_return > 0 else -0.05
        else:
            reward = -0.05

    return reward



def compute_reward_enc_long_trades(action, current_price, previous_price, open_price, position_open, steps_in_trade=0, trades_today=0):
    # ENCOURAGE LONGER PROFITABLE TRADES
    reward = 0.0
    price_trend = current_price - previous_price

    # Parameters
    max_hold_penalty = -0.02
    max_trades_per_day = 5
    overtrade_penalty = -0.02
    hold_penalty_per_step = -0.001
    min_profitable_hold_steps = 10
    profit_duration_bonus = 0.05

    if action == 0:  # HOLD
        if not position_open:
            reward = -0.01 if price_trend > 0 else 0.01
        else:
            trade_return = (current_price - open_price) / open_price
            hold_time_penalty = min(steps_in_trade * hold_penalty_per_step, abs(max_hold_penalty))
            reward = trade_return + hold_time_penalty

    elif action == 1:  # OPEN
        if not position_open:
            reward = 0.0
            if trades_today >= max_trades_per_day:
                reward += overtrade_penalty
        else:
            reward = -0.05

    elif action == 2:  # CLOSE
        if position_open:
            trade_return = (current_price - open_price) / open_price
            reward = trade_return
            if trade_return > 0:
                reward += 0.05  # profit bonus
                if steps_in_trade >= min_profitable_hold_steps:
                    reward += profit_duration_bonus  # bonus for holding profitable trade
            else:
                reward -= 0.05  # penalty for losing trade
        else:
            reward = -0.05

    return reward


# Reward function
def compute_reward_live(action, current_price, previous_price, open_price, position_open, steps_in_trade=0, trades_today=0):
    reward = -0.002
    profit = 0.0
    price_trend = current_price - previous_price
    trading_cost = 0.0
    # Calculate the step profit and rate of change.
    step_profit      = 100 * (current_price - previous_price)/previous_price
    step_profit_sma1 = 100 * (current_price - previous_price)/previous_price
    unrealized_profit_rate_of_change = 0.0


    if action == 0:  # HOLD
        if not position_open:
            # No current trade so check if we're in uptrend to penalize holding
            if step_profit_sma1 < 0:
                reward = 0.0001
            else:
                reward = -0.001
        else:
            # If trade is still open, update nax_profit based on unrealized profit
            unrealized_profit = 100 * (current_price - open_price)/open_price
            unrealized_profit_sma1 = 100 * (current_price - open_price)/open_price
            #max_profit = max(max_profit, unrealized_profit)
            reward = step_profit_sma1 if unrealized_profit_sma1 > 0 else + unrealized_profit_sma1

    elif action == 1:  # OPEN
        if not position_open:
            reward = 0.0001
        else:
            # Penalize opening a trade while already in a trade
            reward = -0.001
    elif action == 2:  # CLOSE
        if not position_open:
            reward = -0.001
        else:
            profit = 100 * (current_price - open_price)/open_price
            #max_profit = max(max_profit, profit)
            reward = profit
            done = True

    return reward


# Reward function
def compute_reward_live2(action, current_price, previous_price, open_price, position_open, steps_in_trade=0, trades_today=0):
    reward = -0.002
    profit = 0.0
    price_trend = current_price - previous_price
    trading_cost = 0.0
    # Calculate the step profit and rate of change.
    step_profit      = 100 * (current_price - previous_price)/previous_price
    step_profit_sma1 = 100 * (current_price - previous_price)/previous_price
    unrealized_profit_rate_of_change = 0.0


    if action == 0:  # HOLD
        if not position_open:
            # No current trade so check if we're in uptrend to penalize holding
            if step_profit_sma1 < 0:
                reward = 0.0001
            else:
                reward = -0.001
        else:
            # If trade is still open, update nax_profit based on unrealized profit
            unrealized_profit = 100 * (current_price - open_price)/open_price
            unrealized_profit_sma1 = 100 * (current_price - open_price)/open_price
            max_profit = max(max_profit, unrealized_profit)
            reward = step_profit_sma1 if unrealized_profit_sma1 > 0 else + unrealized_profit_sma1

    elif action == 1:  # OPEN
        if not position_open:
            reward = 0.0001
        else:
            # Penalize opening a trade while already in a trade
            reward = -0.001
    elif action == 2:  # CLOSE
        if not position_open:
            reward = -0.001
        else:
            profit = 100 * (current_price - open_price)/open_price
            max_profit = max(max_profit, profit)
            reward = profit
            done = True

    return reward


@click.command()
@click.option('--algo', default=1, type=int, show_default=True, help='Specify target algorithm: 1-basic, 2-adjusted for freq, 3-encourage long trades, 4-live')
def main(algo: int):
    # Run simulations and collect rewards
    data_path = './signal_with_market_data.p'
    df = pd.read_pickle(data_path)
    #print(df.columns)

    discount_factor = 0.999
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    # Simulate a few trade episodes and how rewards evolve over time
    episodes = [
        # (actions, prices) per episode
        {
            "name": f"Losing trade {datetime(2024,7,3,17, tzinfo=timezone.utc)}",
            "actions": [1] + [0]*12 + [2],  # OPEN, HOLD x12, CLOSE
            "start": datetime(2024,7,3,17, tzinfo=timezone.utc),
            "duration": timedelta(hours=15),
            "prices": [],
        },
        {
            "name": f"Losing trade {datetime(2024,9,1, tzinfo=timezone.utc)}",
            "actions": [1, 0, 2],  # OPEN, HOLD, CLOSE
            "start": datetime(2024,9,1, tzinfo=timezone.utc),
            "duration": timedelta(hours=6),
            "prices": [],
        },
        {
            "name": f"Winning trade {datetime(2024,9,18,5, tzinfo=timezone.utc)}",
            "actions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HOLD x10 without opening
            "start": datetime(2024,9,18,5, tzinfo=timezone.utc),
            "duration": timedelta(hours=23),
            "prices": [],
        },
        {
            "name": f"Winning trade {datetime(2024,9,26,10, tzinfo=timezone.utc)}",
            "actions": [1, 0, 2, 1, 0, 2, 1, 0, 2],  # OPEN, CLOSE x3
            "start": datetime(2024,9,26,10, tzinfo=timezone.utc),
            "duration": timedelta(hours=9),
            "prices": [],
        },
        {
            "name": f"Winning trade {datetime(2025,1,15,7, tzinfo=timezone.utc)}",
            "actions": [1, 0, 2, 1, 0, 2, 1, 0, 2],  # OPEN, CLOSE x3
            "start": datetime(2025,1,15,7, tzinfo=timezone.utc),
            "duration": timedelta(hours=15),
            "prices": [],
        },
    ]
    #print(df.columns)
    #exit(0)

    # Fill in prices & actions for each episode
    for episode in episodes:
        prices = df[episode['start']:episode['start']+episode['duration']]["s1"].values
        duration = len(prices)
        actions = [1] + [0] * (duration - 2) + [2]  # OPEN, HOLD x(duration-2), CLOSE
        episode["prices"] = prices
        episode["actions"] = actions

    algos = [compute_reward_basic, compute_reward_trade_freq, compute_reward_enc_long_trades, compute_reward_live]
    all_rewards = {}
    print(f"Algo: {algos[algo-1].__name__}")
    for episode in episodes:
        rewards = []
        position_open = False
        open_price = 0.0
        steps_in_trade = 0
        trades_today = 0
        prev_price = 0.0
        current_price = 0.0

        actions = episode["actions"]
        prices = episode["prices"]

        for i in range(len(actions)):
            action = actions[i]
            prev_price = prices[i-1]
            current_price = prices[i]

            if action == 1 and not position_open:
                open_price = current_price
                steps_in_trade = 0
                trades_today += 1

            reward = algos[algo-1](
                        action, 
                        current_price, 
                        prev_price, 
                        open_price, 
                        position_open,
                        steps_in_trade, 
                        trades_today
                     )
            rewards.append(reward)

            if position_open:
                steps_in_trade += 1
            
            # Set position_open after acting
            if not position_open:
                position_open = action == 1
        
        rewards = np.array(rewards)
        discounts = np.array([discount_factor**i for i in range(len(rewards))])
        discounted_rewards = rewards * discounts
        total_rewards = discounted_rewards.sum()
        all_rewards[episode["name"]] = dict(
            rewards=rewards, 
            total_rewards=total_rewards, 
            discounted_rewards=discounted_rewards
        )

    # Plotting
    #plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(len(all_rewards.items()), 1, figsize=(12, 8), sharex=True)
    for idx, item in enumerate(all_rewards.items()):
        name, rewards = item
        #plt.plot(range(1, len(rewards)+1), rewards, marker='o', label=name)
        axs[idx].plot(range(1, len(rewards['rewards'])+1), rewards['rewards'], label="Rewards", color=colors[idx])
        axs[idx].plot(range(1, len(rewards['discounted_rewards'])+1), rewards['discounted_rewards'], marker='.', label="Discounted", color=colors[idx])
        axs[idx].set_title(f"{name}. Total reward: {rewards['total_rewards']:9,.4f}")
        axs[idx].set_ylabel("Reward")
        axs[idx].grid(True)
        axs[idx].legend()
    #plt.title("Reward Evolution per Step for Simulated Episodes")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    #plt.grid(True)
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    main()



