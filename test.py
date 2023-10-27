from splendorMAENV.env import splendor_ma_env


env : splendor_ma_env.SplendorMAEnv = splendor_ma_env.SplendorMAEnv()
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    mask = env.get_mask(agent)
    print(f'agent : {agent} reward : {reward}')
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask= mask)

    env.step(action)
env.close()