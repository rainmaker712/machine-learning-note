import gym

def base_learning(env, num_episodes=100, max_steps_per_episode=100):

    for i_episode in range(num_episodes):
        observation = env.reset()
        states = []
        for t in range(max_steps_per_episode):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            states.append(observation)
                    
            if done:
                print(states)
                print(observation, reward, done, info)
                print(f"Episode finished after {t+1} timesteps")
                break
    
    #         if done and observation == 15:
    # #             env.render()
    #             print(states)
    #             print(observation, reward, done, info)
    #             print(f"Episode finished after {t+1} timesteps")
                
    #             observation = env.reset()
    #             for move in states:
    #                 env.render()
    #                 observation, reward, done, info = env.step(move)
                
    env.close()

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
}