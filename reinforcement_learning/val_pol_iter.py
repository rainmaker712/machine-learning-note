import gym
import numpy as np

from time import time
from utils import *
import matplotlib.pyplot as plt

class Iteration(object):
    
    def __init__(self, env_name):
        
        self.env = gym.make(env_name)        
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.policy = np.zeros((1, self.num_states))
        self.value_list = np.zeros((1, self.num_states))
        
    def val_iter(self, gamma=0.9, max_episodes=1e3):
        
        start = time()

        prev_value_list = self.value_list.copy()

        for episode in range(max_episodes):
            for state in range(self.num_states):
                val = 0
                for action in range(self.num_actions):
                    total_val = 0
                    for prob, new_state, reward, done in env.P[state][action]:
                        value_new_state = prev_value_list[0][new_state]
                        tmp = 0
                        if done:
                            tmp = reward
                        else:
                            tmp = reward + gamma*value_new_state
                        total_val += tmp*prob 
                            
                    if total_val > val:
                        val = total_val
                        self.policy[0][state] = action
                        self.value_list[0][state] = val
            
            prev_value_list = self.value_list.copy()
            
        end = time()
        duration = round(end - start, 4)
        return self.policy[0], episode, duration

    def pol_iter(self, gamma=0.9, max_episodes=1e3):

        start = time()
        
        policy = np.random.randint(self.num_actions, size=(1, self.num_states))
        episode = 0
        
        ## 2
        policy_stable = False
        while not policy_stable:
            episode += 1
            eps = 0
            for state in range(self.num_states):
                value = self.value_list[0][state]
                action = policy[0][state]
                total_val_new_state = 0
                for prob, new_state, reward, done in self.env.P[state][action]:
                    value_new_state = self.value_list[0][new_state]
                    total_value = 0
                    if done:
                        total_value = reward                     
                    else:
                        total_value = reward + gamma*value_new_state
                    total_val_new_state += total_value*prob 
                self.value_list[0][state] = total_val_new_state
                eps = max(eps, np.abs(value-self.value_list[0][s]))

            policy_stable = True
            for state in range(self.num_states):
                old_action = policy[0][state]
                max_value = 0
                for action in range(self.num_actions):
                    total_value = 0
                    for prob, new_state, reward, done in self.env.P[state][action]:
                        value_new_state = self.value_list[0][new_state]
                        total_value = 0
                        if done:
                            total_value = reward
                        else:
                            total_value = reward + gamma*value_new_state
                        total_value += prob*total_value
                    if total_value > max_value:
                        max_value = total_value
                        policy[0][state] = action

                if old_action != policy[0][state]:
                    policy_stable = False
        
        end = time()
        duration = round(end - start, 4)
        return policy[0], episode, duration

    def test_policy(self, policy, episode_limit=10000):

        rewards = []
        episode_counts = []

        for _ in range(episode_limit):
            curr_state = self.env.reset()
            cnt_episode = 0
            done = False
            reward_episode = 0

            while not done and cnt_episode < 500:
                cnt_episode += 1
                action = int(policy[curr_state])
                new_state, reward, done, _ = self.env.step(action)
                reward_episode += reward
                curr_state = new_state
            
            rewards.append(reward_episode)
            episode_counts.append(cnt_episode)

        reward_mean = sum(rewards) / len(rewards)
        episode_mean = sum(episode_counts) / len(episode_counts)

        return reward_mean, episode_mean, rewards, episode_counts

def plot(x, val_y, pol_y, title, ylabel):

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()

    ax.plot(x, val_y, marker="s", color="blue", label="value_iter")
    ax.plot(x, pol_y, marker="o", color="red", label="policy_iter")
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("Discount Rate")
    plt.ylabel(ylabel)
    plt.savefig(title + ".png", dpi=150)
    plt.close()


if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    # env_name = "FrozenLake8x8-v1"
    # env_name = "Taxi-v3"
    # env_name = "CliffWalking-v0"
    
    # base_learning(env)
    
    model_iteration = Iteration(env_name)

    gamma_range = list(np.arange(0, 1.0, 0.1))
    additional_gamma = [0.95, 0.99, 0.999]
    gamma_range = gamma_range + additional_gamma

    vi_pol_list, vi_iter_list, vi_time_list = [], [], []
    vi_mr_list, vi_me_list = [], []

    for gamma in gamma_range:
        vi_policy, vi_solve_iter, vi_solve_time = model_iteration.val_iter(gamma)
        vi_mean_rewards, vi_mean_eps, reward, eps = model_iteration.test_policy(vi_policy)

        vi_pol_list.append(vi_policy)
        vi_iter_list.append(vi_solve_iter)
        vi_time_list.append(vi_solve_time)
        vi_mr_list.append(vi_mean_rewards)
        vi_me_list.append(vi_mean_eps)
    
    pi_pol_list, pi_iter_list, pi_time_list = [], [], []
    pi_mr_list, pi_me_list = [], []

    for gamma in gamma_range:
        pi_policy, pi_solve_iter, pi_solve_time = model_iteration.pol_iter(gamma)
        pi_mean_rewards, pi_mean_eps, _, _ = test_policy(env, pi_policy)
        
        pi_pol_list.append(pi_policy)
        pi_iter_list.append(pi_solve_iter)
        pi_time_list.append(pi_solve_time)
        pi_mr_list.append(pi_mean_rewards)
        pi_me_list.append(pi_mean_eps)

    print("score:", np.max(vi_mr_list), np.max(pi_mr_list))
    print("trial:", np.mean(vi_iter_list), np.mean(pi_iter_list))
    print("time:", np.mean(vi_time_list), np.mean(pi_time_list))

    plot(gamma_range, vi_mr_list, pi_mr_list, f"Mean rewards for {env_name}", "Scores")
    plot(gamma_range, vi_iter_list, pi_iter_list, f"Mean episodes for {env_name}", "Episodes")
    plot(gamma_range, vi_time_list, pi_time_list, f"Mean time for {env_name}", "Time")