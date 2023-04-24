# draw charts for score.csv file

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv", header=None)
df.columns = ["episode", "loss", "q_value", "reward", "epsilon"]
df["moving_average_reward"] = df["reward"].rolling(100).mean()
df["moving_average_loss"] = df["loss"].rolling(100).mean()
df["moving_average_q_value"] = df["q_value"].rolling(100).mean()
df["moving_average_epsilon"] = df["epsilon"].rolling(100).mean()
df["episode"] = df.index + 1
# df.to_csv("results.csv", index=False)

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(df["episode"], df["moving_average_reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.subplot(2, 2, 2)
plt.plot(df["episode"], df["moving_average_loss"])
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.subplot(2, 2, 3)
plt.plot(df["episode"], df["moving_average_q_value"])
plt.xlabel("Episode")
plt.ylabel("Q Value")
plt.subplot(2, 2, 4)
plt.plot(df["episode"], df["moving_average_epsilon"])
plt.xlabel("Episode")
plt.ylabel("Epsilon")
# plt.savefig("results.png")
plt.show()


# read scores.csv file
# import pandas as pd

df = pd.read_csv("scores.csv", header=None)
df.columns = ["score"]
# add new column for episode
df["episode"] = df.index + 1

# draw charts for score.csv file
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(df["episode"], df["score"])
plt.xlabel("Episode")
plt.ylabel("Score")
plt.subplot(2, 2, 2)
plt.plot(df["episode"], df["score"].rolling(100).mean())
plt.xlabel("Episode")
plt.ylabel("Moving Average Score")
plt.show()
# q: what is moving average score?
# a: moving average score is the average of the last 100 scores