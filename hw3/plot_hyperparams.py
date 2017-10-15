import matplotlib.pyplot as plt

point_one_x_filename = 'ram_output_lr_01x.txt'
one_x_filename = 'ram_output_lr_1x.txt'
five_x_filename = 'ram_output_lr_5x.txt'
ten_x_filename = 'ram_output_lr_10x.txt'
hundred_x_filename = 'ram_output_lr_100x.txt'

filenames = [point_one_x_filename, one_x_filename, five_x_filename, ten_x_filename, hundred_x_filename]
labels = ['0.1x', '1x', '5x', '10x', '100x']

for label, filename in zip(labels, filenames):
    with open(filename) as f:
        mean_reward = []
        timesteps = []
        for line in f:
            if 'Timestep' in line:
                timesteps.append(int(line.split()[-1]))
            if 'mean reward (100 episodes)' in line:
                mean_reward.append(float(line.split()[-1]))
    plt.plot(timesteps, mean_reward, label=label)
plt.legend()
plt.ylabel('mean 100-episode reward')
plt.xlabel('timesteps')
plt.show()
