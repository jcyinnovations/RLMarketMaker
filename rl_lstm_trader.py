import pandas as pd
import os 
from datetime import datetime, timedelta, timezone
#import gym
#from gym import spaces
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import click
from getpass import getpass
import json 

from sb3_contrib import RecurrentPPO  # Requires sb3_contrib package
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Callback to capture new, per-step reward metric
class RewardPerStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPerStepCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # The Monitor wrapper adds an "episode" key in the info dict at the end of an episode.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                reward_per_step = episode_reward / episode_length if episode_length > 0 else 0
                # Log the metric (for example, to TensorBoard)
                self.logger.record("rollout/reward_per_step", reward_per_step)
                if self.verbose:
                    print("Episode reward per step:", reward_per_step)
        return True


# Ensure your DataFrame contains a 'price' column for calculating trade profit.
# If necessary, adjust the column name in the environment code below.

# Step 2. Define the custom Gym trading environment.
class TradingEnv(gym.Env):
    def __init__(self, data, trading_cost=0.0, render_mode='human'):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.trading_cost = trading_cost
        self.lambda_drawdown = 0.75  # Penalty weighting for maximum drawdown.
        self.target_duration = 24  # Maximum trade duration in hours.
        # Action space: 0 = hold, 1 = open trade, 2 = close trade.
        self.action_space = gym.spaces.Discrete(3)
        # Observation space: each row of data plus current position flag.
        num_features = self.data.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features + 1,), dtype=np.float32)
        self.num_envs = 16
        #self.observation_space = spaces.Box(low=-1000000.0, high=1000000.0, shape=(num_features + 1,), dtype=np.float32)
        self.position = 0  # 0: not in a trade, 1: in a trade.
        self.entry_price = 0.0  # Price at which trade is opened.
        self.prev_price = 0.0  # Previous price for calculating hold returns
        self.max_profit = 0.0
        self.trade_start_step = 0  # Step at which trade is opened.
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.position = 0
        self.entry_price = 0.0
        self.prev_price = 0.0
        self.max_profit = 0.0
        self.trade_start_step = 0
        #self.current_step = 0
        self.current_step = np.random.randint(0, len(self.data) - self.target_duration)
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Combine current row's features with the position indicator.
        state = self.data.iloc[self.current_step].values
        #print(f"State: {state} ||")
        return np.append(state, self.position)
    
    def step(self, action):
        reward = -0.0001  # Default reward for not taking any action.
        done = False
        truncated = False
        
        # Get current price from the DataFrame.
        current_price = self.data.iloc[self.current_step]['price']
        # Process the action.
        #print(f"----->Step: {self.current_step:9,}, Action: {action}")
        if action == 0:  # Hold position.
            # Action 0 (Hold) gives:
            # - a small time penalty if not in a trade
            # - a small reward if in a trade based on current return
            reward = -0.0001
            # Now calculate the hold reward (discounted profit/loss of current step)
            if self.prev_price == 0.0:
                step_profit = 0.0
            else:
                step_profit = 100 * (current_price - self.prev_price)/self.prev_price
            if self.position == 1:
                # If trade is still open, update nax_profit based on unrealized profit
                trade_duration = self.current_step - self.trade_start_step + 1
                unrealized_profit = 100 * (current_price - self.entry_price - self.trading_cost)/self.entry_price
                self.max_profit = max(self.max_profit, unrealized_profit)
                time_penalty = self.time_penalty(trade_duration)
                # Holding gets penalized over long periods of time
                reward = step_profit * time_penalty + ((1-time_penalty) * -0.1)
                log_message = {
                    'side': 'hold',
                    'current_price': current_price,
                    'step': self.current_step,
                    'profit': step_profit,
                    'reward': reward,
                    'max_profit': self.max_profit,
                    'time_penalty': time_penalty
                }
                print(json.dumps(log_message))
            else:
                # No current trade so check if we're in uptrend to penalize holding
                if step_profit < 0:
                    reward = 0.0001
        elif action == 1:  # Open trade.
            if self.position == 0:
                reward = 0.0001
                log_message = {
                    'side': 'open',
                    'current_price': current_price,
                    'step': self.current_step,
                    'reward': reward
                }
                print(json.dumps(log_message))
                self.position = 1
                self.entry_price = current_price
                self.trade_start_step = self.current_step
                self.max_profit = 0.0
        elif action == 2:  # Close trade.
            if self.position == 1:
                trade_duration = self.current_step - self.trade_start_step + 1
                profit = 100 * (current_price - self.entry_price - self.trading_cost)/self.entry_price
                self.max_profit = max(self.max_profit, profit)
                drawdown_penalty = self.lambda_drawdown * (self.max_profit - (current_price - self.entry_price))
                # Final drawdown penalty based on maximum drawdown experienced:
                time_penalty = self.time_penalty(trade_duration)
                reward = (profit * time_penalty ) - drawdown_penalty
                log_message = {
                    'side': 'close',
                    'current_price': current_price,
                    'step': self.current_step,
                    'profit': profit,
                    'reward': reward,
                    'max_profit': self.max_profit,
                    'trade_duration': trade_duration,
                    'drawdown_penalty': drawdown_penalty,
                    'time_penalty': time_penalty
                }
                print(json.dumps(log_message))
                done = True
                #self.position = 0
                #self.entry_price = 0.0
                #self.max_profit = 0.0
                #self.prev_price = 0.0

        self.prev_price = current_price
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.current_step = np.random.randint(0, len(self.data) - self.target_duration)
            #print("Done:", self.current_step, len(self.data))
            # Episode ends due to end of data
            #if self.current_step >= len(self.data) - 1:
            #    truncated = True
            # Set new starting point for the next episode.
            done = True
        
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, truncated, {}
    
    def time_penalty(self, current_duration):
        # Shape the discount to keep trades shorted than 24 hours
        discount = 1.0
        if current_duration > self.target_duration:
            discount = self.target_duration/current_duration
        return discount

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}")


#################################################################
#  LSTM-based Policy for Trading with Stable Baselines 3 (SB3)
#################################################################

@click.command()
@click.option('--timesteps', default=500000, type=int, show_default=True, help='Run-length in number of timesteps')
@click.option('--iteration', default=3, type=int, show_default=True, help='Current Iteration')
@click.option('--discount_factor', default=0.999, type=float, show_default=True, help='Discount Factor') 
@click.option('--eval_frequency', default=100000, type=float, show_default=True, help='Frequency of evaluations') 
@click.option('--checkpoint_frequency', default=10000, type=int, show_default=True, help='Frequency of checkpoints') 
def main(timesteps: int, iteration: int, discount_factor: float, eval_frequency: int, checkpoint_frequency: int):
    #iteration = 3
    #total_timesteps = 2000000
    iteration_name = f"iteration-{iteration}"
    models_dir = f"./models/{iteration_name}/"
    logdir = f"./logs/{iteration_name}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Step 1. Load your pickled DataFrame.
    data_path = './signals_df_2024_v40_iteration1_epoch206-TO_20250225-2-LONG.p'
    df = pd.read_pickle(data_path)

    # Quick inspection of the data.
    print("DataFrame head:")
    print(df.head())
    print("\nDataFrame columns:", df.columns.tolist())

    # Remove columns not necessary for inference
    df.drop(columns=['date', 'ground_truth', 'pnl'], inplace=True)

    # Step 3. Instantiate the environment, wrapped with DummyVecEnv, then VecNormalize
    train_env = DummyVecEnv([lambda: TradingEnv(df, trading_cost=0.1)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    initial_obs = train_env.reset()
    print("\nInitial observation:", initial_obs)
    # Step 4. Create and train the Recurrent PPO model using an LSTM-based policy.
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=logdir,
        gamma=discount_factor,
        learning_rate=0.0001,
    )
    # target_kl=0.5,

    # Setup Eval environment similar to training
    eval_env = TradingEnv(df, trading_cost=0.1)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False,
        training=False, 
        clip_obs=10., 
        gamma=discount_factor
    )
    # Configure the evaluation callback.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logdir,
        eval_freq=eval_frequency, 
        n_eval_episodes=5,
        deterministic=True,
    )
    # Configure a Checkpoint callback to save the model frequently since
    # Evalcallback only saves the best model based on the mean reward.
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency, 
        save_path=f"{models_dir}/checkpoints/",
        name_prefix="rppo_trading_model"
    )

    # Reward per step callback
    rewards_per_step_callback = RewardPerStepCallback(verbose=1)

    start = datetime.now(timezone.utc)
    print("######################")
    print("#   Training Start.  #")
    print(f"# {start} #")
    print("######################")
    model.learn(
        total_timesteps=timesteps, 
        callback=[eval_callback, checkpoint_callback, rewards_per_step_callback]
    )
    model.save(f"{models_dir}/rppo_trading_model")
    # Save the VecNormalize statistics for the recurrent model.
    train_env.save(f"{models_dir}/vec_normalize_env_rnn.pkl")

    end = datetime.now(timezone.utc)
    print("######################")
    print("# Training complete. #")
    print(f"# {end} #")
    print("######################")

    # Step 5. Evaluate the trained agent.
    obs = train_env.reset()
    # Recurrent policies require initial state and episode_start flag.
    recurrent_states = None  # initialize hidden states as None
    episode_start = np.array([True])  # marks the beginning of an episode
    rewards = []
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()