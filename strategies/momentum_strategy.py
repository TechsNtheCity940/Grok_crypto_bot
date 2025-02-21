from stable_baselines3 import PPO

class MomentumStrategy:
    def __init__(self, model_path='models/trained_models/ppo_trading_model'):  # Kept as is assuming absolute path
        self.model = PPO.load(model_path)
        
    def add_indicators(df):
        df['rsi'] = compute_rsi(df['close'], window=14)
        return df

    def get_action(self, observation):
        action, _ = self.model.predict(observation)
        return action  # 0: hold, 1: buy, 2: sell