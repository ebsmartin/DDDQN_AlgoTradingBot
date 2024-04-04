import matplotlib.pyplot as plt
import json


class DataVisualizer:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        with open ("Output/episode_mem.json", "r") as f:
            self.episode_data = json.load(f)
        with open ("Output/testing_mem.json", "r") as f:
            self.testing_data = json.load(f)

    def plot_portfolio_value(self, data, title, eth_price, filename):
        plt.figure(figsize=(20, 12))
        plt.plot(eth_price["Close"][::50], label="ETH Price")
        plt.plot(data[len(data)-1]["Portfolio Value"][::50], label="Portfolio Value")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.title(title)
        plt.savefig(filename)

    def plot_portfolio_value_test(self, data, title, eth_price, filename):
        plt.figure(figsize=(20, 12))
        plt.plot(eth_price["Close"][::10], label="ETH Price")
        plt.plot(data["Portfolio Value"][::10], label="Portfolio Value")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.title(title)
        plt.savefig(filename)

    def plot_inventory_held(self, data, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(data["Inventory Size"][::10], label="Inventory Held")
        plt.xlabel("Time Steps")
        plt.ylabel("Amount")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(filename)

    def plot_actions(self, data, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(data["Actions"][::10], label="Actions")
        plt.xlabel("Time Steps")
        plt.ylabel("Action")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(filename)

    def visualize_data(self):
        selected_episode = len(self.episode_data) - 1
        self.plot_portfolio_value(self.episode_data, f"Portfolio Value for Training", self.train_data, "Output/episode_portfolio.png")
        self.plot_portfolio_value_test(self.testing_data, f"Portfolio Value for Testing", self.test_data, "Output/testing_portfolio.png")
