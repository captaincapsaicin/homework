import matplotlib.pyplot as plt

filenames = ['output.txt']
labels = ['mean return over time']

for label, filename in zip(labels, filenames):
    with open(filename) as f:
        mean_reward = []
        timesteps = []
        best_mean_reward = []
        for line in f:
            if 'Timestep' in line:
                timesteps.append(int(line.split()[-1]))
            if 'mean reward (100 episodes)' in line:
                mean_reward.append(float(line.split()[-1]))
            if 'best mean reward' in line:
                best_mean_reward.append(float(line.split()[-1]))
    plt.plot(timesteps, mean_reward, label='mean 100-episode reward')
    plt.plot(timesteps, best_mean_reward, label='best mean reward')
plt.legend()
plt.ylabel('reward')
plt.xlabel('timesteps')
plt.show()
