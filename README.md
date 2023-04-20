# Deep Reinforcement-Based Algorithmic Trading Agent:
## An attempt to create a profitable trading bot using Google search trend history, price data, and sentiment if feasible.

### Eric Burton Martin
_Colorado State University_

1. **INTRODUCTION:**

Neural networks and artificial intelligence have fascinated me ever since I learned about their existence and now after learning about reinforcement learning I am eager to apply the skills I will learn throughout this semester to make an algorithmic trading bot. As of 2017, I have been interested in cryptocurrency, trading, and blockchain technology, so I intend to combine my newfound skills in machine learning with my knowledge of trading to create an algorithmic trading bot that utilizes reinforcement learning with access to price data, sentiment analysis, and google search trend values. I recently took a Python for Financial Analysis Course taught by the VP of Quantitative Analysis at Wells Fargo and I am eager to put some of the methods and knowledge I have learned to the test. Alongside that, I am interested in trading bots because they seem like a project that can truly pay off if enough effort is put into them (but I am not holding my breath). I am not expecting a profitable or applicable bot, but proof-of-concept is a start. To utilize the final product, I will have to connect the bot to the Coinbase-Pro API which I already have a license for. I can also test the bot in their Sandbox environment which will allow for safe real-time and historical testing.

2. **KEYWORDS:**

Artificial Intelligence, Reinforcement Learning, Algorithmic Trading, Machine Learning

3. **RELATED WORK:**

There are a plethora of related works that I shall be exploring although none have used real-time search trend history as an input from what I have seen. I will be adding specific sources as the project progresses.

4. **Goal:**

The goal of this project is to create a trading bot that can learn and adapt to the stock/crypto markets through reinforcement learning. The agent will be able to analyze past trading data and use it to make informed decisions about stock/crypto trading. The agent will use the reinforcement learning techniques to learn from its mistakes and hopefully improve its trading performance over time.

5. **METHODOLOGY:**

The methods and algorithms I will consider using in this project and their reasonings are listed below. Note that I will most likely end up only using search trend and price data since this project will already be proving difficult.

_Search Query Trends:_
- Google is the main platform for search queries so a good method of looking into the public’s current concerns and interests is through search trends.
- Utilizing the unofficial Google Trends API called pytrends a SearchTrends class will be created that will be able to obtain trend data of various timeframes.

_Training Environments:_
- OpenAI's Gym is a toolkit for developing and testing reinforcement learning algorithms and it has a variety of simulated stock trading environments that can be used to train AI agents.
- If OpenAI’s Gym proves difficult I will look into other stock trading training environments like Q-Trader, Quantopian, and TradingGym.
- I have also found some helpful YouTube videos showing examples of using these playgrounds.

_Coinbase Pro API:_
- Coinbase is one of the largest cryptocurrency exchanges in the United States and has an API that allows for trading bot-based trades.
- Coinbase also allows for safe trading in their real-time sandbox environment.

_Correlation Matrices and Granger Causality:_
- Since I will be supplying the network with a vast amount of input data, determining which of these inputs are actually useful for our bot will be important as it can allow me to trim the inputs to inputs that show correlation or imply causation.
- The main method I will use to determine correlations
