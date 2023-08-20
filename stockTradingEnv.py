import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float16)
        # 3 features: balance, stock_held, stock_price; to be added: sentiment_score, moving_average, change shape to no. of features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float16)
        
        # load stock price data
        self.df = df
        self.current_step = 0
        
        # initialize balance, stocks held, etc
        self.balance = 10000
        self.stock_held = 0
        self.stock_price = 0
        self.net_worth = self.balance
        
    def step(self, action):
        # execute one time step within env
        self.current_step += 1
        self.stock_price = self.df.loc[self.current_step, 'Close']
        max_buying_power = max(self.balance*0.1, min(self.balance, 100))
        
        # Implement Trading Logic
        # HOLD: requires no logice (change nothing)
        action_val = action[0]
        if action_val > 0:
            # buy
            spend_amt = action_val * max_buying_power
            shares_to_buy = spend_amt / self.stock_price
            self.balance -= spend_amt
            self.stock_held += shares_to_buy
            
        elif action_val < 0:
            # sell
            shares_to_sell = abs(action_val) * self.stock_held
            self.balance += shares_to_sell * self.stock_price
            self.stock_held -= shares_to_sell
        
        
        # Calculate Reward and net worth
        self.net_worth = self.balance + self.stock_held * self.stock_price
        reward = self.net_worth - self.balance
        
        # check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # get info
        info = {'balance': self.balance, 'stock held': self.stock_held}
        
        return self._next_observation(), reward, done, info
    
    def reset(self):
        # reset state of environment to initial state
        self.balance = 10000
        self.stock_held = 0
        self.stock_price = self.df.loc[self.current_step, 'Close']
        self.net_worth = self.balance
        self.current_step = 0
        
        return self._next_observation()
    
    def get_net_worth(self):
        return self.net_worth
    
    def _next_observation(self):
        # get the next observation
        # TODO: implement below methods
        # sentiment_score = self.get_sentiment_score()
        # moving_average = self.calculate_moving_average()
        obs = np.array([self.balance, self.stock_held, self.stock_price])
        return obs
    
    def render(self, mode='human', close=False):
        # render the env to screen (in this case, print)
        print(f'Step: {self.current_step}, Balance: {self.balance}, Stocks held: {self.stock_held}, Net Worth: {self.net_worth}')
            