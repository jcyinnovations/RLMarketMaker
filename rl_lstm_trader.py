import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO  # Requires sb3_contrib package
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os 
from datetime import datetime, timedelta, timezone

# Ensure your DataFrame contains a 'price' column for calculating trade profit.
# If necessary, adjust the column name in the environment code below.

# Step 2. Define the custom Gym trading environment.
class TradingEnv(gym.Env):
    def __init__(self, data, trading_cost=0.0):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.trading_cost = trading_cost
        self.current_step = 0
        
        # Action space: 0 = hold, 1 = open trade, 2 = close trade.
        self.action_space = spaces.Discrete(3)
        
        # Observation space: each row of data plus current position flag.
        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features + 1,), dtype=np.float32)
        
        self.position = 0  # 0: not in a trade, 1: in a trade.
        self.entry_price = 0.0  # Price at which trade is opened.
        
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_observation()
    
    def _get_observation(self):
        # Combine current row's features with the position indicator.
        state = self.data.iloc[self.current_step].values
        return np.append(state, self.position)
    
    def step(self, action):
        reward = 0.0
        done = False
        
        # Get current price from the DataFrame.
        current_price = self.data.iloc[self.current_step]['price']
        
        # Process the action.
        #print(f"Step: {self.current_step:9,}")
        if action == 1:  # Open trade.
            if self.position == 0:
                print(f"Opening trade at price: {current_price:9,.2f}, step: {self.current_step}")
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Close trade.
            if self.position == 1:
                profit = current_price - self.entry_price - self.trading_cost
                print(f"Closing trade at price: {current_price:9,.2f}, step: {self.current_step}. Profit: {profit:9,.2f}")
                reward = profit
                self.position = 0
                self.entry_price = 0.0
        # Action 0 (Hold) gives no reward (or could include a small time penalty).
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}")


#################################################################
#  LSTM-based Policy for Trading with Stable Baselines 3 (SB3)
#################################################################

if __name__ == "__main__":
    iteration = 3
    total_timesteps = 2000000
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

    # Step 3. Instantiate the environment.
    env = TradingEnv(df, trading_cost=0.1)
    # Wrap the environment with DummyVecEnv and then VecNormalize.
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    initial_obs = env.reset()
    print("\nInitial observation:", initial_obs)

    print("######################")
    print("#   Training Start.  #")
    print(f"# {datetime.now(timezone.utc)} #")
    print("######################")

    # Step 4. Create and train the Recurrent PPO model using an LSTM-based policy.
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=logdir
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{models_dir}/rppo_trading_model")
    # Save the VecNormalize statistics for the recurrent model.
    env.save(f"{models_dir}/vec_normalize_env_rnn.pkl")

    print("######################")
    print("# Training complete. #")
    print(f"# {datetime.now(timezone.utc)} #")
    print("######################")

    # Step 5. Evaluate the trained agent.
    obs = env.reset()
    # Recurrent policies require initial state and episode_start flag.
    recurrent_states = None  # initialize hidden states as None
    episode_start = np.array([True])  # marks the beginning of an episode
    rewards = []

    while True:
        # Pass hidden state and episode_start flag to predict.
        action, recurrent_states = model.predict(
            obs, 
            state=recurrent_states, 
            episode_start=episode_start, 
            deterministic=True
        )
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        env.render()
        # After the first step, episode_start becomes False.
        episode_start = np.array([False])
        if done:
            break

    print(f"\nTotal reward from evaluation: {sum(rewards)}")
    print("########################")
    print("# Evaluation complete. #")
    print("########################")
    env.close()