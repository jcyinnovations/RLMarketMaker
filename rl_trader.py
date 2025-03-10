import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# IMPORTANT: Ensure that your DataFrame contains a column named 'price'.
# If the column has a different name, adjust the code accordingly.

# Step 2. Build a custom Gym environment.
class TradingEnv(gym.Env):
    def __init__(self, data, trading_cost=0.0):
        super(TradingEnv, self).__init__()
        self.data = data #data.reset_index(drop=True)
        self.trading_cost = trading_cost
        self.current_step = 0
        
        # Define action space:
        # 0: Hold, 1: Open trade, 2: Close trade
        self.action_space = spaces.Discrete(3)
        
        # Define observation space:
        # Here, we assume each row in the DataFrame contains the network's output probability 
        # along with additional features. We append the current position (0 or 1) to these features.
        num_features = self.data.shape[1]  # number of columns in your DataFrame
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features + 1,), dtype=np.float32)
        
        self.position = 0  # 0: not in a trade, 1: in a trade
        self.entry_price = 0.0  # record the price at which a trade is opened
        
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_observation()
    
    def _get_observation(self):
        # Construct the state by taking the row of features and appending the position.
        state = self.data.iloc[self.current_step].to_numpy()
        return np.append(state, self.position)
    
    def step(self, action):
        reward = 0.0
        done = False
        
        # Retrieve the current market price from the DataFrame.
        current_price = self.data.iloc[self.current_step]['price']
        
        # Execute the selected action.
        if action == 1:  # Open trade
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Close trade
            if self.position == 1:
                # Calculate profit: (current price - entry price) adjusted by trading costs.
                profit = current_price - self.entry_price - self.trading_cost
                reward = profit
                self.position = 0
                self.entry_price = 0.0
        # If action == 0 (Hold), then reward remains 0 (or you could add a small penalty or time cost).
        
        # Move to the next time step.
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # Get the next state.
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}")



# Step 1. Load your pickled DataFrame.
data_path = './signals_df_2024_v40_iteration1_epoch206-TO_20250225-2-LONG.p'
df = pd.read_pickle(data_path)

# Quick inspection of the data.
print("DataFrame head:")
print(df.head())
print("\nDataFrame columns:", df.columns.tolist())

# Remove columns not necessary for inference
df.drop(columns=['date', 'ground_truth', 'pnl'], inplace=True)

# Step 3. Instantiate the environment.
env = TradingEnv(df, trading_cost=0.1)
# Wrap the environment with DummyVecEnv and then VecNormalize.
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

initial_obs = env.reset()
print("\nInitial observation:", initial_obs)

# Step 4. Train an RL model using PPO.
# You can adjust total_timesteps based on your data and training needs.
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5000000)
model.save("ppo_trading_model")

# Save the VecNormalize statistics for future use.
env.save("vec_normalize_env.pkl")

# Optional: Evaluate the trained agent.
obs = env.reset()
rewards = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    env.render()
    if done:
        break

print("\nTotal reward from evaluation:", sum(rewards))
