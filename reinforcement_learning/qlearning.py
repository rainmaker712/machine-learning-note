import gym
import numpy as np
import random

from time import time
from utils import *
import matplotlib.pyplot as plt

class QLearning(object):

    def __init__(self, env_name):
        
        self.env = gym.make(env_name)        
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.gamma = 0.9
        self.decay_rate = None
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def runner(self, epsilon=0.1, learning_rate=0.1, total_eps=1e3):

        start = time()
        rewards = []
        episode_counts = []

        for episode in range(total_eps):
            curr_state = self.env.reset()
            cnt_episode = 0
            done = False
            reward_episode = 0

            while not done and cnt_episode < 500:
                cnt_episode += 1
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[curr_state])
                new_state, reward, done, _ = self.env.step(action)
                reward_episode += reward
                curr_state = new_state

                if not done:
                    self.q_table[curr_state, action] = self.q_table[curr_state, action] + learning_rate*(reward + self.gamma*np.max(self.q_table[new_state, :]) - self.q_table[curr_state, action])
                else:
                    self.q_table[curr_state, action] = self.q_table[curr_state, action] + learning_rate*(reward - self.q_table[curr_state,action])

            rewards.append(reward_episode)
            episode_counts.append(cnt_episode)

        reward_mean = sum(rewards) / len(rewards)
        episode_mean = sum(episode_counts) / len(episode_counts)

        end = time()
        duration = round(end - start, 4)

        return np.argmax(self.q_table, axis=1), episode_mean, duration, self.q_table, rewards

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

        


def q_learning(env, min_epsilon=0.01, learning_rate=0.1, decay_rate=None,
               total_episodes=1e5, discount=0.9, ):
    
    start = time()
    
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    
    qtable = np.zeros((number_of_states, number_of_actions))
    gamma = discount

    # exploration parameter
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    
    if not decay_rate:
        decay_rate = 1./total_episodes
    
    rewards = []
    for episode in range(int(total_episodes)):
        # reset the environment
        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        while True:

            # choose an action a in the corrent world state
            exp_exp_tradeoff = random.uniform(0,1)

            # if greater than epsilon --> exploit
            if exp_exp_tradeoff > epsilon:
                b = qtable[state, :]
                action = np.random.choice(np.where(b == b.max())[0])
#                 action = np.argmax(qtable[state, :])
            # else choose exploration
            else:
                action = env.action_space.sample()

            # take action (a) and observe the outcome state (s') and reward (r)    
            new_state, reward, done, info = env.step(action)
            total_reward += reward
            # update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max(Q (s', a') - Q(s,a))]
            if not done:
                qtable[state, action] = qtable[state, action] + learning_rate*(reward + gamma*np.max(qtable[new_state, :]) - qtable[state, action])
            else:
                qtable[state, action] = qtable[state,action] + learning_rate*(reward - qtable[state,action])

            # change state
            state = new_state

            # is it Done
            if done:
                break
                
        # reduce epsilon 
        rewards.append(total_reward)
        epsilon = max(max_epsilon -  decay_rate * episode, min_epsilon) 
    #     print (epsilon)
    
    end = time()
    time_spent = round(end - start, 4)
    print("Solved in: {} episodes and {} seconds".format(total_episodes, time_spent))
    return np.argmax(qtable, axis=1), total_episodes, time_spent, qtable, rewards




if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    # env_name = "FrozenLake8x8-v1"
    # env_name = "Taxi-v3"
    # env_name = "CliffWalking-v0"
    # base_learning(env)
    
    q_learning = QLearning(env_name)

    lr_list = [0.1,0.3,0.5,0.9]
    epilson_list = [0.5,0.7, 0.9]

    q_map_dict = {}
    for lr in lr_list:
        q_map_dict[lr] = {}
        for eps in epilson_list:
            q_map_dict[lr][eps] = {}
            q_policy, q_iter, q_time, q_table, rewards = q_learning.runner(eps, lr)
            q_avg_reward, q_avg_eps, _, __ = q_learning.test_policy(q_policy)

            q_map_dict[lr][eps]["avg_reward"] = q_avg_reward
            q_map_dict[lr][eps]["avg_eps"] = q_avg_eps
            q_map_dict[lr][eps]["q-table"] = q_table
            q_map_dict[lr][eps]["rewards"] = rewards 
            q_map_dict[lr][eps]["iteration"] = q_iter
            q_map_dict[lr][eps]["duration"] = q_time
            q_map_dict[lr][eps]["policy"] = q_policy

    the_dict = q_map_dict

    import pandas as pd
    the_df = pd.DataFrame(columns=["Average eps", "Learning Rate", "Epsilon", "Reward", "Duration"])

    for lr in lr_list:
        for eps in q_map_dict[lr]:
            rew = q_map_dict[lr][eps]["avg_reward"]
            duration = q_map_dict[lr][eps]["duration"]
            meps = q_map_dict[lr][eps]["avg_eps"]

            dic = {"Average eps": meps, "Learning Rate":lr, 
                "Epsilon": eps, "Reward": rew, "Time Spent": duration}        
            the_df = the_df.append(dic, ignore_index=True)

    lr_reward_0 = the_df[the_df["Epsilon"] == epilson_list[0]]["Reward"]
    lr_reward_1 = the_df[the_df["Epsilon"] == epilson_list[1]]["Reward"]
    lr_reward_2 = the_df[the_df["Epsilon"] == epilson_list[2]]["Reward"]

    title = f"Q-Learning - learning rate, {env_name}"
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.plot(lr_list, lr_reward_0, marker="s", color="blue", label="epilson=0.5")
    ax.plot(lr_list, lr_reward_1, marker="o", color="red", label="epilson=0.7")
    ax.plot(lr_list, lr_reward_2, marker=".", color="green", label="epilson=0.9")
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("learning rate")
    plt.ylabel("score")
    plt.savefig(title + ".png", dpi=150)
    plt.close()

    ep_reward_0 = the_df[the_df["Learning Rate"] == lr_list[0]]["Reward"]
    ep_reward_1 = the_df[the_df["Learning Rate"] == lr_list[1]]["Reward"]
    ep_reward_2 = the_df[the_df["Learning Rate"] == lr_list[2]]["Reward"]
    ep_reward_3 = the_df[the_df["Learning Rate"] == lr_list[3]]["Reward"]

    title = f"Q-Learning - epilson, {env_name}"
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.plot(epilson_list, ep_reward_0, marker="s", color="blue", label="lr=0.1")
    ax.plot(epilson_list, ep_reward_1, marker="o", color="red", label="lr=0.3")
    ax.plot(epilson_list, ep_reward_2, marker=".", color="green", label="lr=0.5")
    ax.plot(epilson_list, ep_reward_3, marker="8", color="yellow", label="lr=0.9")
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("epsilon")
    plt.ylabel("score")
    plt.savefig(title + ".png", dpi=150)
    plt.close()


    the_df["Learning Rate"]
    the_df["Epsilon"]
    the_df["Reward"]
    the_df["Time Spent"]
    the_df["Training Means_eps"]

    the_df.to_csv("q_learning_results_8x8.csv")
    # "0": left, "1": down, "2": right, "3": up