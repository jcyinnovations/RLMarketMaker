import pandas as pd
import os 
from datetime import datetime, timedelta, timezone

import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import click
from getpass import getpass
import json 
import math 

from sb3_contrib import RecurrentPPO  
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from CustomEvalCallback import CustomEvalCallback
from stable_baselines3.common.utils import set_random_seed

class TimeSeriesCNNExtractor(BaseFeaturesExtractor):
    '''
    Custom feature extractor for time series data using a CNN.
    '''
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_features, seq_len = observation_space.shape  # (12, L)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Reduce time dim to 1
            nn.Flatten()
        )

        self._features_dim = 64  # Output size of last layer

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch, 12, L)
        return self.cnn(observations)


# Callback to capture new, per-step reward metric
class RewardPerStepCallback(BaseCallback):
    '''
    Custom callback to log the reward per step at the end of each episode.
    This is useful for tracking the performance of the agent over time.
    '''
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
    '''
    Custom trading environment for reinforcement learning.
    '''
    ACTION_HOLD = 0
    ACTION_OPEN = 1
    ACTION_CLOSE = 2
    SIDE = ['hold', 'open', 'close']

    def __init__(self, data, trading_cost=0.0, max_duration=None, is_eval=False, render_mode='human'):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.trading_cost = trading_cost
        self.lambda_drawdown = 1.25  # Penalty weighting for maximum drawdown.
        # Action space: 0 = hold, 1 = open trade, 2 = close trade.
        self.action_space = gym.spaces.Discrete(3)
        # Observation space: each row of data plus current position flag.
        num_features = self.data.shape[1] + 1 # Data columns plus position flag
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        #self.num_envs = 16
        self.position = 0  # 0: not in a trade, 1: in a trade.
        self.entry_price = 0.0  # Price at which trade is opened.
        self.prev_price = 0.0  # Previous price for calculating hold returns
        self.max_profit = 0.0
        self.trade_start_step = 0  # Step at which trade is opened.
        self.current_step = 0
        # Time limit for the environment during learning.
        self.max_duration = max_duration
        self.is_eval = is_eval

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state.
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        self.position = 0
        self.trade_duration = 0
        self.entry_price = 0.0
        self.entry_sma1 = 0.0
        self.prev_price = 0.0
        self.prev_sma1 = 0.0
        self.prev_sma4 = 0.0
        self.max_profit = 0.0
        self.trade_start_step = 0
        self.current_duration = 0

        self.unrealized_profit = 0.0
        self.unrealized_profit_sma1 = 0.0

        self.step_profit_sma1 = 0.0
        self.step_profit_sma4 = 0.0
        self.step_profit = 0.0

        hold_back = self.max_duration if self.max_duration else 0
        if self.is_eval:
            # Continue from last step for next epoch
            self.current_step += 1
        else:
            # Randomized starting point for next epoch
            self.current_step = np.random.randint(0, len(self.data) - hold_back - 1)

        if self.current_step >= len(self.data) - 1:
            self.current_step = 0

        return self._get_observation(), {}
    
    def _get_observation(self):
        '''
        Combine current row's features with the position indicator.
        '''
        state = self.data.iloc[self.current_step].values
        return np.append(state, self.position).astype(np.float32)
    
    def compute_reward(self, action, trades_today=0):
        position_open = self.position == 1

        reward = -0.002
        trading_cost = 0.0

        if action == 0:  # HOLD
            if not position_open:
                # No current trade so check if we're in uptrend to penalize holding
                #if self.step_profit_sma1 < 0:
                #    reward = 0.0001
                #else:
                #    reward = -0.001
                reward = -0.0001
            else:
                # If in a trade, reward is based on unrealized profit
                reward = self.step_profit_sma1 #self.unrealized_profit_sma1
                # Penalize for holding too long
                if self.trade_duration > 0 and self.max_profit > 0 and self.unrealized_profit_sma1>0 and self.max_profit > self.unrealized_profit_sma1:
                    reward = reward - self.lambda_drawdown * (self.max_profit - self.unrealized_profit_sma1)
        elif action == 1:  # OPEN
            if not position_open:
                reward = 0.0001
            else:
                # Penalize opening a trade while already in a trade
                reward = -5.001
        elif action == 2:  # CLOSE
            if not position_open:
                # Penalize closing a trade while not in a trade
                reward = -5.001
            else:
                reward = self.unrealized_profit_sma1 * 1.10

        return reward
    

    def step(self, action):
        '''
        Execute one time step within the environment.
        '''
        done = False
        truncated = False
        ####################################
        # UPDATE PRIOR TO CALCULATING REWARD
        ####################################
        current_price = self.data.iloc[self.current_step]['c']
        current_sma1 = self.data.iloc[self.current_step]['s1']
        current_sma4 = self.data.iloc[self.current_step]['s4']
        # Only applies after reset
        if self.prev_price == 0.0:
            if self.current_step > 0:
                self.prev_price = self.data.iloc[self.current_step - 1]['c']
                self.prev_sma1 = self.data.iloc[self.current_step - 1]['s1']
                self.prev_sma4 = self.data.iloc[self.current_step - 1]['s4']
            else:
                self.prev_price = current_price
                self.prev_sma1 = current_sma1
                self.prev_sma4 = current_sma4
        # Calculate the step profit and rate of change.
        self.step_profit      = 100 * (current_price - self.prev_price)/self.prev_price
        self.step_profit_sma1 = 100 * (current_sma1 - self.prev_sma1)/self.prev_sma1
        self.step_profit_sma4 = 100 * (current_sma4 - self.prev_sma4)/self.prev_sma4

        if self.position == 1:
            # If trade is still open, update nax_profit based on unrealized profit
            self.trade_duration = self.current_step - self.trade_start_step + 1
            self.unrealized_profit = 100 * (current_price - self.entry_price - self.trading_cost)/self.entry_price
            self.unrealized_profit_sma1 = 100 * (current_sma1 - self.entry_sma1 - self.trading_cost)/self.entry_sma1
            self.max_profit = max(self.max_profit, self.unrealized_profit)
        else:
            self.unrealized_profit = 0.0
            self.unrealized_profit_sma1 = 0.0

        ####################################
        # CALCULATE REWARD
        ####################################
        reward = self.compute_reward(
            action, 
            trades_today=0
        )

        ####################################
        # UPDATE AFTER ACTION
        ####################################
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.entry_sma1 = current_sma1
            self.trade_start_step = self.current_step
            self.max_profit = 0.0
            self.trade_duration = 0
            self.unrealized_profit = 0.0
            self.unrealized_profit_sma1 = 0.0

        if self.is_eval:
            log_message = {
                'side': TradingEnv.SIDE[action],
                'current_price': current_price,
                'step': self.current_step,
                'unrealized_profit': self.unrealized_profit,
                'step_profit': self.step_profit_sma1,
                'reward': reward,
                'max_profit': self.max_profit,
                'trade_duration': self.trade_duration,
            }
            print(json.dumps(log_message))

        self.prev_price = current_price
        self.prev_sma1 = current_sma1
        self.prev_sma4 = current_sma4
        self.current_step += 1
        self.current_duration += 1

        if self.position == 1 and action == 2:
            # Close the trade
            done = True

        if self.current_step >= len(self.data) - 1:
            truncated = True

        if self.max_duration:
            if self.current_duration >= self.max_duration:
                truncated = True
            
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        info = {}
        if done:
            info = dict(
                is_success=self.unrealized_profit>0,
                step=self.current_step,
                profit=self.unrealized_profit,
                max_profit=self.max_profit
            )
        elif truncated:
            info = dict(
                is_success=self.unrealized_profit>0,
                step=self.current_step,
                profit=self.unrealized_profit,
                max_profit=self.max_profit
            )
        if self.is_eval and (done or truncated):
            print(f"|--->Episode finished after {self.current_step} steps. Profit: {self.unrealized_profit:.2f}%, Max Profit: {self.max_profit:.2f}%")

        return next_state, reward, done, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}")


#################################################################
#  LSTM-based Policy for Trading with Stable Baselines 3 (SB3)
#################################################################
@click.command()
@click.option('--timesteps', default=500000, type=int, show_default=True, help='Run-length in number of timesteps')
@click.option('--iteration', default=3, type=int, show_default=True, help='Current Iteration')
@click.option('--discount_factor', default=0.99, type=float, show_default=True, help='Discount Factor') 
@click.option('--eval_frequency', default=100000, type=float, show_default=True, help='Frequency of evaluations') 
@click.option('--checkpoint_frequency', default=10000, type=int, show_default=True, help='Frequency of checkpoints') 
@click.option('--parallel_envs', default=64, type=int, show_default=True, help='Number of parallel environments') 
def main(timesteps: int, iteration: int, discount_factor: float, eval_frequency: int, checkpoint_frequency: int, parallel_envs: int):
    iteration_name = f"iteration-{iteration}"
    models_dir = f"./models/{iteration_name}/"
    logdir = f"./logs/{iteration_name}/"
    max_duration = 576  # Max duration for each episode (in steps)
    trading_cost = 0.0  # Trading cost (e.g., commission, slippage)
    clip_obs = 1.0  # Clipping value for observations
    print(f"Training Iteration: {iteration_name} with {parallel_envs} parallel environments and discount factor {discount_factor}.")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Step 1. Load your pickled DataFrame.
    data_path = './signal_with_market_data.p'
    df = pd.read_pickle(data_path)

    # Step 2. Instantiate the environment, wrapped with DummyVecEnv, then VecNormalize
    train_env = DummyVecEnv([lambda: TradingEnv(df, trading_cost=trading_cost, max_duration=max_duration) for _ in range(parallel_envs)])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=clip_obs
    )
    initial_obs = train_env.reset()
    
    # Learning Rate
    def linear_schedule(initial_lr: float):
        def schedule(progress_remaining: float):
            return initial_lr * progress_remaining
        return schedule


    # Step 3. Create and train the Recurrent PPO model using an LSTM-based policy.
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=logdir,
        gamma=discount_factor,
        learning_rate=0.0002, #linear_schedule(0.0002),
        clip_range=0.2,
        n_steps=max_duration,
        gae_lambda=0.90,
        #n_epochs=128,
        #batch_size=1024,
        policy_kwargs=dict(
            net_arch=dict(vf=[512,128,32], pi=[512,128,32]),
            lstm_hidden_size=512,
            n_lstm_layers=1,
            enable_critic_lstm=True,
        ),
    )

    # Setup Eval environment similar to training
    eval_env = TradingEnv(df, trading_cost=trading_cost, is_eval=True, max_duration=576, render_mode='human')
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False,
        training=False, 
        clip_obs=clip_obs, 
    )
    # Configure the evaluation callback.
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logdir,
        eval_freq=eval_frequency, 
        n_eval_episodes=200,
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
    print("#####################################")
    print("########   Training Start.  #########")
    print(f"# {start} #")
    print("#####################################")
    model.learn(
        total_timesteps=timesteps, 
        callback=[eval_callback, checkpoint_callback, rewards_per_step_callback]
    )
    model.save(f"{models_dir}/rppo_trading_model")
    # Save the VecNormalize statistics for the recurrent model.
    train_env.save(f"{models_dir}/vec_normalize_env_rnn.pkl")

    end = datetime.now(timezone.utc)
    print( "#####################################")
    print( "######## Training complete. #########")
    print(f"# {end} #")
    print( "#####################################")

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
