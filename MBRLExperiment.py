#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


def run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate, gamma, policy,
                    n_planning_updates, wind_proportion):
    print("Running repetitions with the following settings:")
    print(locals())
    n_evals = n_timesteps // eval_interval + 1
    eval_returns = np.zeros((n_repetitions, n_evals))
    for rep in range(n_repetitions):
        print(f"Running repitition {rep+1}", end="\r")
        env = WindyGridworld(wind_proportion)
        if policy == 'dyna':
            pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif policy == 'ps':
            pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)
        s = env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            pi.update(s, a, r, done, s_next, n_planning_updates)

            if done:
                s = env.reset()
            else:
                s = s_next

            if t % eval_interval == 0:
                eval_num = t // eval_interval
                eval_env = WindyGridworld(wind_proportion)
                eval_return = pi.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100)
                eval_returns[rep, eval_num] = eval_return

    mean_eval_returns = np.mean(eval_returns, axis=0)
    return mean_eval_returns


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updates = [1, 3, 5]

    # IMPLEMENT YOUR EXPERIMENT HERE

    smoothing_window = 6
    eval_timesteps = [i*eval_interval for i in range(n_timesteps // eval_interval + 1)]

    q_learning_baseline = {
        wp: run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate,
                            gamma, policy='dyna', n_planning_updates=0, wind_proportion=wp) for wp in wind_proportions}

    policy = 'dyna'
    for wp in wind_proportions:
        plot = LearningCurvePlot(f"Dyna Learning Curves (wind_proportion={wp})")
        plot.add_curve(eval_timesteps, smooth(q_learning_baseline[wp], smoothing_window), label="q_learning")
        for n_pu in n_planning_updates:
            eval_returns = run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate, gamma,
                                           policy, n_planning_updates=n_pu, wind_proportion=wp)
            plot.add_curve(eval_timesteps, smooth(eval_returns, smoothing_window), label=f"n_planning_updates = {n_pu}")
        plot.save(name=f"dyna_wp_{wp}.png")

    policy = 'ps'
    for wp in wind_proportions:
        plot = LearningCurvePlot(f"Prioritized sweeping Learning Curves (wind_proportion={wp})")
        plot.add_curve(eval_timesteps, smooth(q_learning_baseline[wp], smoothing_window), label="q_learning")
        for n_pu in n_planning_updates:
            eval_returns = run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate, gamma,
                                           policy, n_planning_updates=n_pu, wind_proportion=wp)
            plot.add_curve(eval_timesteps, smooth(eval_returns, smoothing_window), label=f"n_planning_updates = {n_pu}")
        plot.save(name=f"ps_wp_{wp}.png")


if __name__ == '__main__':
    experiment()
