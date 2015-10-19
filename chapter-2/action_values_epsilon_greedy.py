from scipy.stats import norm
from numpy.random import randint, uniform

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

import time

def cumulative_moving_average(current_average, current_plays, next_value):
    return (next_value + (current_plays * current_average)) / (current_plays + 1)


class Arm(object):
    def __init__(self, qstar):
        self.qstar = qstar

    def reward(self):
        return norm.rvs(loc=self.qstar, scale=1)


class Bandit(object):
    def __init__(self, qstars):
        self.arms = []
        self.optimal_mean_reward = max(qstars)
        for i, qstar in enumerate(qstars):
            self.arms.append(Arm(qstar))
            if qstar == self.optimal_mean_reward:
                self.optimal_arm = i

    def reward(self, arm):
        return self.arms[arm].reward()


class Agent(object):
    def __init__(self, bandit, epsilon=0.1):
        self.epsilon = epsilon
        self.bandit = bandit
        self.num_bandit_arms = len(self.bandit.arms)
        self.current_plays = [0] * self.num_bandit_arms
        self.estimated_values = [0.0] * self.num_bandit_arms

    def get_greedy_action(self):
        return self.estimated_values.index(max(self.estimated_values))

    def get_random_action(self):
        return randint(self.num_bandit_arms)

    def choose_action(self):
        return self.get_greedy_action() if uniform() >= self.epsilon else self.get_random_action()

    def perform_action(self, action):
        reward = self.bandit.reward(action)
        self.estimated_values[action] = cumulative_moving_average(self.estimated_values[action],
                                                                  self.current_plays[action],
                                                                  reward)
        self.current_plays[action] += 1

        return reward


class Task(object):
    def __init__(self, num_plays=1000, agent=None, bandit=None):
        self.num_plays = num_plays
        self.agent = agent if agent else Agent
        self.bandit = bandit if bandit else Bandit(norm.rvs(size=10))
        self.agent = agent if agent else Agent(bandit)
        self.rewards = []
        self.optimal_action_tally = 0.0
        self.optimal_action_percentages = []

    def execute(self):
        for i in range(1, self.num_plays + 1):
            action = self.agent.choose_action()
            self.rewards.append(self.agent.perform_action(action))
            if action == self.bandit.optimal_arm:
                self.optimal_action_tally += 1
            self.optimal_action_percentages.append(self.optimal_action_tally / i * 100)
        return self.rewards


class Experiment(object):
    def __init__(self, num_tasks=2000, epsilons=[0.0, 0.01, 0.1]):
        self.epsilons = epsilons
        self.bandits = [Bandit(norm.rvs(size=10)) for _ in range(num_tasks)]
        self.tasks = {epsilon: [Task(agent=Agent(bandit, epsilon=epsilon), bandit=bandit)
                                for bandit in self.bandits]
                      for epsilon in epsilons}
        self.rewards = {epsilon: [] for epsilon in epsilons}

    def execute(self):
        for epsilon, tasks in self.tasks.iteritems():
            for task in tasks:
                self.rewards[epsilon].append(task.execute())

    def get_average_rewards(self, epsilon):
        return [np.mean(group) for group in zip(*self.rewards[epsilon])]

    def get_average_optimal_action_percentage(self, epsilon):
        return [np.mean(group) for group in zip(*[task.optimal_action_percentages
                for task in self.tasks[epsilon]])]


def main():
    start_time = time.time()

    print 'Creating experiment'
    experiment = Experiment(num_tasks=2000)

    print 'Executing experiment'
    experiment.execute()

    print 'Evaluating experiment'
    optimal_mean_rewards = [bandit.optimal_mean_reward for bandit in experiment.bandits]

    mean = np.mean(optimal_mean_rewards)
    variance = np.var(optimal_mean_rewards)
    sigma = np.sqrt(variance)

    print 'Preparing graphs'
    fig = plt.figure()

    # Mean optimal reward

    ax1 = fig.add_subplot('311')
    plt.title('Optimal reward mean distribution')

    x1 = np.linspace(min(optimal_mean_rewards), max(optimal_mean_rewards), 1000)
    ax1.plot(x1, mlab.normpdf(x1, mean, sigma))

    ax1.hist(optimal_mean_rewards, normed=True, histtype='stepfilled', alpha=0.2, bins=50)

    ax1.axvline(mean)

    # Mean reward value

    ax2 = fig.add_subplot('312')
    plt.title('Average rewards')
    plt.xticks(np.arange(0, 1001, 250))
    line1, = ax2.plot(range(1, 1001), experiment.get_average_rewards(0.0), label='0.0')
    line2, = ax2.plot(range(1, 1001), experiment.get_average_rewards(0.01), label='0.01')
    line3, = ax2.plot(range(1, 1001), experiment.get_average_rewards(0.1), label='0.1')
    ax2.legend(handles=[line1, line2, line3], loc='center left', bbox_to_anchor=(1, 0.5))

    # Mean optimal action percentage

    ax3 = fig.add_subplot('313')
    plt.title('Average optimal action percentage')
    plt.xticks(np.arange(0, 1001, 250))
    line4, = ax3.plot(range(1, 1001), experiment.get_average_optimal_action_percentage(0.0), label='0.0')
    line5, = ax3.plot(range(1, 1001), experiment.get_average_optimal_action_percentage(0.01), label='0.01')
    line6, = ax3.plot(range(1, 1001), experiment.get_average_optimal_action_percentage(0.1), label='0.1')
    ax3.legend(handles=[line4, line5, line6], loc='center left', bbox_to_anchor=(1, 0.5))

    print "Time elapsed: {} seconds".format(time.time() - start_time)

    plt.show()


if __name__ == '__main__':
    main()
