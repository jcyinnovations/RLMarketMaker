import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO  # Requires sb3_contrib package
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os 
import click
from datetime import datetime, timedelta, timezone
from rl_lstm_trader import TradingEnv
import json 

# Ensure your DataFrame contains a 'price' column for calculating trade profit.
# If necessary, adjust the column name in the environment code below.

#################################################################
#  LSTM-based Policy for Trading with Stable Baselines 3 (SB3)
#################################################################

@click.command()
@click.option('--iteration', default=5, type=int, show_default=True, help='Current Iteration')
@click.option('--checkpoint', default=1000000, type=int, show_default=True, help='Target Checkpoint for Evaluation') 
def main(iteration: int, checkpoint: int):
    print("Evaluation for Iteration:", iteration, " Checkpoint:", checkpoint)
    #iteration = 3
    iteration_name = f"iteration-{iteration}"
    models_dir = f"./models/{iteration_name}/"

    # Step 1. Load your pickled DataFrame.
    data_path = './signals_df_2024_v40_iteration1_epoch206-TO_20250225-2-LONG.p'
    df = pd.read_pickle(data_path)
    # Remove columns not necessary for inference
    df.drop(columns=['date', 'ground_truth', 'pnl'], inplace=True)

    # Assume real_world_data is your new DataFrame with real-world signals.
    env_real = TradingEnv(df, trading_cost=0.1, render_mode='human')
    env_real = DummyVecEnv([lambda: env_real])
    # Load the normalization parameters saved during training.
    env_real = VecNormalize.load(f"{models_dir}/vec_normalize_env_rnn.pkl", env_real)
    # Make sure to disable further normalization updates.
    env_real.training = False
    env_real.norm_reward = False

    initial_obs = env_real.reset()
    #print("\nInitial observation:", initial_obs)

    #print("Loading Model...")
    #model = RecurrentPPO.load(f"{models_dir}/rppo_trading_model")
    #model = RecurrentPPO.load(f"{models_dir}/best_model")
    model = RecurrentPPO.load(f"{models_dir}/checkpoints/rppo_trading_model_{checkpoint}_steps")

    # Step 5. Evaluate the trained agent.
    obs = env_real.reset()

    # Recurrent policies require initial state and episode_start flag.
    recurrent_states = None  # initialize hidden states as None
    episode_start = np.array([True])  # marks the beginning of an episode
    rewards = []
    episode_rewards = []
    #print("Run evals...")
    rounds = 0
    while True:
        # Pass hidden state and episode_start flag to predict.
        action, recurrent_states = model.predict(
            obs, 
            state=recurrent_states, 
            episode_start=episode_start, 
            deterministic=True
        )
        obs, reward, done, _ = env_real.step(action)
        rewards.append(reward[0])
        episode_rewards.append(reward[0])
        env_real.render()
        # After the first step, episode_start becomes False.
        episode_start = np.array([False])
        rounds += 1
        if done:
            print({"rewards": episode_rewards})
            recurrent_states = None  # reset hidden states when an episode ends
            obs = env_real.reset()
            episode_start = np.array([True])
            episode_rewards = []
            #print("Done. Rounds:", rounds)
        if rounds > 10000:
            break
    '''
    print(f"Rounds: {rounds}")
    print(f"\nTotal reward from evaluation:", sum(rewards))
    print("########################")
    print("# Evaluation complete. #")
    print("########################")'
    '''
    env_real.close()

if __name__ == "__main__":
    main()