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
        env = WindyGridworld(wind_proportion)  # Initialise a clean environment
        if policy == 'dyna':  # Initialise a clean agent
            pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif policy == 'ps':
            pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)
        else:
            raise ValueError("Invalid policy type given")
        s = env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, epsilon)  # Sample action
            s_next, r, done = env.step(a)  # Simulate environment
            pi.update(s, a, r, done, s_next, n_planning_updates)
            if done:  # Reset when environment terminates
                s = env.reset()
            else:
                s = s_next
            # Greedy evaluation after each interval
            if t % eval_interval == 0:
                eval_num = t // eval_interval
                eval_env = WindyGridworld(wind_proportion)
                eval_return = pi.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100)
                eval_returns[rep, eval_num] = eval_return
    # Compute average evaluations over all repetitions
    mean_eval_returns = np.mean(eval_returns, axis=0)
    return mean_eval_returns


def experiment(dyna=True, ps=True, comp=True):
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updates = [1, 3, 5]

    # List of evaluation timesteps to use on the x axis of the plots
    eval_timesteps = [i*eval_interval for i in range(n_timesteps // eval_interval + 1)]
    smoothing_window = 7

    # Run the Q-Learning baseline for both wind proportions
    q_learning_baseline = {
        wp: run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate,
                            gamma, policy='dyna', n_planning_updates=0, wind_proportion=wp) for wp in wind_proportions}

    ###########
    # 1: Dyna #
    ###########

    dyna_eval_returns = {wp: {} for wp in wind_proportions}
    if dyna:
        policy = 'dyna'
        for wp in wind_proportions:
            plot = LearningCurvePlot(f"Dyna Learning Curves (wind_proportion={wp})")
            # Add the Q-Learning baseline curve
            plot.add_curve(eval_timesteps, smooth(q_learning_baseline[wp], smoothing_window), label="q_learning")
            for n_pu in n_planning_updates:
                eval_returns = run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate, gamma,
                                               policy, n_planning_updates=n_pu, wind_proportion=wp)
                dyna_eval_returns[wp][n_pu] = eval_returns
                plot.add_curve(eval_timesteps, smooth(eval_returns, smoothing_window),
                               label=f"n_planning_updates = {n_pu}")
            plot.save(name=f"dyna_wp_{wp}.png")

    ###########################
    # 1: Prioritized sweeping #
    ###########################

    ps_eval_returns = {wp: {} for wp in wind_proportions}
    if ps:
        policy = 'ps'
        for wp in wind_proportions:
            plot = LearningCurvePlot(f"Prioritized sweeping Learning Curves (wind_proportion={wp})")
            # Add the Q-Learning baseline curve
            plot.add_curve(eval_timesteps, smooth(q_learning_baseline[wp], smoothing_window), label="q_learning")
            for n_pu in n_planning_updates:
                eval_returns = run_repetitions(n_repetitions, n_timesteps, eval_interval, epsilon, learning_rate, gamma,
                                               policy, n_planning_updates=n_pu, wind_proportion=wp)
                ps_eval_returns[wp][n_pu] = eval_returns
                plot.add_curve(eval_timesteps, smooth(eval_returns, smoothing_window),
                               label=f"n_planning_updates = {n_pu}")
            plot.save(name=f"ps_wp_{wp}.png")

    #################
    # 3: Comparison #
    #################

    if dyna and ps and comp:
        for wp in wind_proportions:
            plot = LearningCurvePlot(f"Comparing Dyna and Prioritized sweeping (wind_proportion={wp})")
            # Add the Q-Learning baseline curve
            plot.add_curve(eval_timesteps, smooth(q_learning_baseline[wp], smoothing_window), label="q_learning")

            best_dyna_auc = 0
            best_dyna = (None, [])
            # Find the best performing model using the area under the learning curve
            for n_pu, eval_return in dyna_eval_returns[wp].items():
                auc = np.trapz(eval_return)
                if auc > best_dyna_auc:
                    best_dyna_auc = auc
                    best_dyna = (n_pu, eval_return)
            plot.add_curve(eval_timesteps, smooth(best_dyna[1], smoothing_window),
                           label=f"Dyna (n_planning_updates = {best_dyna[0]})")

            best_ps_auc = 0
            best_ps = (None, [])
            # Find the best performing model using the area under the learning curve
            for n_pu, eval_return in ps_eval_returns[wp].items():
                auc = np.trapz(eval_return)
                if auc > best_ps_auc:
                    best_ps_auc = auc
                    best_ps = (n_pu, eval_return)
            plot.add_curve(eval_timesteps, smooth(best_ps[1], smoothing_window),
                           label=f"Prioritized sweeping (n_planning_updates = {best_ps[0]})")
            plot.save(name=f"comp_wp_{wp}.png")


if __name__ == '__main__':
    experiment()
