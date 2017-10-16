import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		states = np.vstack([state]*self.num_simulated_paths)
		states_per_step = []
		actions_per_step = []
		next_states_per_step = []
		for i in range(self.horizon):
			actions = np.vstack([self.env.action_space.sample() for i in range(self.num_simulated_paths)])
			next_states = self.dyn_model.predict(states, actions)

			# append to set
			states_per_step.append(states)
			actions_per_step.append(actions)
			next_states_per_step.append(next_states)

			states = next_states

		states_per_step = np.array(states_per_step)
		actions_per_step = np.array(actions_per_step)
		next_states_per_step = np.array(next_states_per_step)

		costs = trajectory_cost_fn(self.cost_fn, states_per_step, actions_per_step, next_states_per_step)
		best_trajectory = np.argmin(costs)
		return actions_per_step[0][best_trajectory]
