# DQN-Stock-Trade

This project leverages Deep Q-Learning (DQN) to create a stock trading bot. It consists of several components that work together to fetch stock data, analyze sentiment, and make trading decisions based on the learned policy.

## Files Overview

### 1. `DQN.py`
This file contains the implementation of the Deep Q-Learning model. DQN is a reinforcement learning algorithm that learns to take actions to maximize cumulative reward. In the context of stock trading, it learns to buy, sell, or hold stocks to maximize profit.

### 2. `DQNAgent.py`
The agent file, where the DQN agent is implemented. An agent is responsible for interacting with the environment, taking actions based on the current state, and learning from the results. It uses the DQN model to decide the best actions and updates the model based on the observed rewards.

### 3. `env.ipynb`
This Jupyter Notebook contains the training loop and the environment where the agent is trained. The environment simulates the stock market, providing the agent with information about stock prices and allowing it to execute trades.

### 4. `sentiment_analysis.ipynb`
This notebook is responsible for fetching news articles related to the stocks and analyzing their sentiment. Although it currently only fetches and analyzes news articles, it can be extended to include more sophisticated sentiment analysis techniques to influence trading decisions.

### 5. `stock_fetcher.ipynb`
This notebook is used to download previous stock data. It can be configured to fetch data for specific stocks and time periods, providing the historical data needed for training and backtesting the model.

### 6. `stockTradingEnv.py`
This file contains the class definition for the stock trading environment. It simulates the dynamics of the stock market, including price changes and trading mechanics. The agent interacts with this environment to learn the optimal trading strategy.

## TODO

- **Enhance Sentiment Analysis**: Plan to add more sophisticated sentiment analysis techniques using news articles to make the model more robust. This will allow the model to consider market sentiment and potentially improve trading decisions.

## Getting Started

1. **Install Dependencies**: Make sure to install all required libraries and dependencies, including:
   - Gym: For creating and managing the stock trading environment.
   - Numpy: For numerical computations.
   - Pandas: For data manipulation and analysis.
   - PyTorch: For building and training the Deep Q-Learning model.
   - yfinance: For retrieving stock data
2. **Configure Parameters**: Set the desired stock symbols, time frames, and other parameters in the respective files.
3. **Run Training**: Execute the `env.ipynb` notebook to train the agent.
4. **Evaluate Performance**: Analyze the performance of the trained agent using various metrics and visualizations.


## Conclusion

This project provides a comprehensive framework for stock trading using DQN. By integrating various components like sentiment analysis and historical data fetching, it offers a robust and extensible platform for experimenting with algorithmic trading.


