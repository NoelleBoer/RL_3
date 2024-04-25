#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.Q_sa = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        self.n_counts = np.zeros((n_states, n_actions, n_states))  # Initialize transition counts with zeros
        self.r_sums = np.zeros((n_states, n_actions, n_states))  # Initialize reward sums with zeros

    def select_action(self, s, epsilon):  # ϵ-greedy policy
        best_action = np.argmax(self.Q_sa[s])
        if np.random.random() > epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Update model
        self.n_counts[s, a, s_next] += 1
        self.r_sums[s, a, s_next] += r

        def q_table_update(s, a, r, s_next):
            a_next = np.argmax(self.Q_sa[s_next])
            td_target = r + self.gamma * self.Q_sa[s_next, a_next]
            td_delta = td_target - self.Q_sa[s, a]
            self.Q_sa[s, a] += self.learning_rate * td_delta

        q_table_update(s, a, r, s_next)

        for _ in range(n_planning_updates):
            # Find state with n(s) > 0
            n_s_candidates = np.argwhere(np.sum(self.n_counts, axis=(1, 2)) > 0).flatten()
            s = np.random.choice(n_s_candidates)
            # Find action with n(s,a) > 0
            n_sa_candidates = np.argwhere(np.sum(self.n_counts[s], axis=1) > 0).flatten()
            a = np.random.choice(n_sa_candidates)
            # Simulate model
            n_sa = self.n_counts[s][a]
            s_next_props = n_sa / np.sum(n_sa)
            s_next = np.random.choice(self.n_states, p=s_next_props)
            r = self.r_sums[s][a][s_next] / self.n_counts[s][a][s_next]

            q_table_update(s, a, r, s_next)

        pass

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff

        self.Q_sa = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        self.n_counts = np.zeros((n_states, n_actions, n_states))  # Initialize transition counts with zeros
        self.r_sums = np.zeros((n_states, n_actions, n_states))  # Initialize reward sums with zeros
        self.queue = PriorityQueue()  # Initialize PQ

    def select_action(self, s, epsilon):  # ϵ-greedy policy
        best_action = np.argmax(self.Q_sa[s])
        if np.random.random() > epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Update model
        self.n_counts[s, a, s_next] += 1
        self.r_sums[s, a, s_next] += r
        # Compute priority
        a_next = np.argmax(self.Q_sa[s_next])
        td_target = r + self.gamma * self.Q_sa[s_next, a_next]
        p = np.abs(td_target - self.Q_sa[s, a])
        if p > self.priority_cutoff:
            # State-action needs update
            self.queue.put((-p, (s, a)))

        for _ in range(n_planning_updates):
            # Sample PG, bream when empty
            if self.queue.empty():
                break
            _, (s, a) = self.queue.get()
            # Simulate model
            n_sa = self.n_counts[s][a]
            s_next_props = n_sa / np.sum(n_sa)
            s_next = np.random.choice(self.n_states, p=s_next_props)
            r = self.r_sums[s][a][s_next] / self.n_counts[s][a][s_next]
            # Update Q-table
            a_next = np.argmax(self.Q_sa[s_next])
            td_target = r + self.gamma * self.Q_sa[s_next, a_next]
            td_delta = td_target - self.Q_sa[s, a]
            self.Q_sa[s, a] += self.learning_rate * td_delta

            for s_top, a_top in np.argwhere(self.n_counts[:, :, s] > 0):
                # Get reward from model
                r_top = self.r_sums[s_top][a_top][s] / self.n_counts[s_top][a_top][s]
                # Compute priority
                a = np.argmax(self.Q_sa[s])
                td_target = r_top + self.gamma * self.Q_sa[s, a]
                p = np.abs(td_target - self.Q_sa[s_top, a_top])
                if p > self.priority_cutoff:
                    # State-action needs update
                    self.queue.put((-p, (s_top, a_top)))

        pass

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps'  # or 'ps'
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_timesteps):
        # Select action, transition, update policy
        a = pi.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)

        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next


if __name__ == '__main__':
    test()
