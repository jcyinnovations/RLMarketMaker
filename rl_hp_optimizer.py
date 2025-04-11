import optuna
from optuna import Trial
import gymnasium
from sb3_contrib import RecurrentPPO  
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from rl_lstm_trader import TradingEnv
import os 
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = 20000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3
N_WARMUP_STEPS = 20


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def setup_environment(iteration=26, training=True, parallel_envs=1, max_duration=1024):
    # Step 1. Load your pickled DataFrame.
    data_path = './signal_with_market_data.p'
    df = pd.read_pickle(data_path)

    if training:
        env = DummyVecEnv([lambda: TradingEnv(df, trading_cost=0.1, max_duration=max_duration)])
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=True, 
            training=training, 
            clip_obs=1.
        )
    else:
        env = TradingEnv(df, trading_cost=0.1, env_name="EVAL")
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=False,
            training=False, 
            clip_obs=1.,
        )

    return env

def setup_dirs(iteration=26):
    """
    Set up directories for saving models and logs.
    """
    iteration_name = f"iteration-{iteration}"
    models_dir = f"./models/{iteration_name}/"
    logdir = f"./logs/{iteration_name}/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return models_dir, logdir

def objective_fn(trial: Trial):
    """
    Objective function to optimize hyperparameters for RL agent.

    Args:
        trial (Trial): Optuna trial object.

    Returns:
        float: The reward obtained from the RL agent.
    """
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 256, step=16)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    discount_factor = trial.suggest_uniform('discount_factor', 0.9, 0.9999)
    n_steps = trial.suggest_int('n_steps', 16, 1024, step=32)
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 512, step=64)

    # Initialize and train the RL agent with the suggested hyperparameters
    # agent = RLAgent(learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)
    # reward = agent.train_and_evaluate()
    # For demonstration purposes, we will return a dummy reward
    iteration = 26
    timesteps = N_TIMESTEPS
    # Set up directories for saving models and logs
    models_dir, logdir = setup_dirs(iteration=iteration)
    max_duration = 1000 # Max duration for each episode (in steps)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train_env = setup_environment(iteration=iteration, training=True, max_duration=max_duration)
    obs = train_env.reset()
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        train_env, 
        verbose=1, 
        tensorboard_log=logdir,
        gamma=discount_factor,
        learning_rate=learning_rate,
        clip_range=0.2,
        n_steps=n_steps,
        policy_kwargs=dict(
            net_arch=dict(vf=[512,128,32], pi=[512,128,32]),
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=1,
            enable_critic_lstm=True,
        ),
    )

    # Configure the evaluation callback.
    eval_env = setup_environment(iteration=iteration, training=False, max_duration=max_duration)
    eval_callback = TrialEvalCallback(
        eval_env = eval_env,
        trial = trial,
        n_eval_episodes = N_EVAL_EPISODES,
        eval_freq = EVAL_FREQ,
        deterministic= True,
        verbose = 1,
    )

    nan_encountered = False
    try:
        # Train the model
        print("Training the model...")
        model.learn(
            total_timesteps=timesteps,
            callback=eval_callback,
        )
    except Exception as e:
        print(f"Exception encountered during training: {e}")
        nan_encountered = True
    finally:
        # Save the model
        model.env.close()
        eval_env.close()

    if nan_encountered:
        return float("nan") # Dummy reward for successful training
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return eval_callback.last_mean_reward


def optimize_hyperparameters(n_trials: int=100) -> None:
    """
    Optimize hyperparameters using Optuna.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_fn, n_trials=n_trials)

    # Print the best hyperparameters and their corresponding reward
    print("Best hyperparameters: ", study.best_params)
    print("Best reward: ", study.best_value)


def optimize_hyperparameters_with_storage(storage: str, N_TRIALS: int=100):
    """
    Optimize hyperparameters using Optuna with a specified storage.
    """
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_WARMUP_STEPS)

    study = optuna.create_study(
        study_name="rl_hyperparameter_optimization", 
        sampler=sampler, 
        pruner=pruner, 
        direction="maximize", 
        storage=storage
    )

    try:
        study.optimize(objective_fn, n_trials=N_TRIALS, timeout=600)
    except Exception as e:
        print(f"Exception encountered during optimization: {e}")

    print("Number of finished trials: ", len(study.trials))
    print("==============================================")
    print("Best trial:")
    trial = study.best_trial

    print("\tValue: ", trial.value)
    print("\tParameters: ")
    print("\t______________")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")

    print("\tUser attrs:")
    print("\t______________")
    for key, value in trial.user_attrs.items():
        print(f"\t\t{key}: {value}")


if __name__ == "__main__":
    # Run the hyperparameter optimization
    #optimize_hyperparameters(n_trials=100)
    optimize_hyperparameters_with_storage(None, N_TRIALS=100)    