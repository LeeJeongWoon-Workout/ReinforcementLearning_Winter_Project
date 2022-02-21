

# Policy Iteration
nx = 5
ny = 5
env = GridworldEnv([ny, nx])
dp_agent = TensorDP(gamma=0.9)
dp_agent.set_env(env)
dp_agent.reset_policy()
info_pi = dp_agent.policy_iteration()
figsize_mul = 10
steps = info_pi['converge']

fig, ax = plt.subplots(nrows=steps, ncols=2, figsize=(steps * figsize_mul, 
                                         figsize_mul * 2))


for i in range(steps):
    visualize_value_function(ax[i][0],info_pi['v'][i], nx, ny)
    visualize_policy(ax[i][1], info_pi['pi'][i], nx, ny) 
    
#Value Iteration
dp_agent.reset_policy()
info_vi = dp_agent.value_iteration(compute_pi=True)
figsize_mul = 10
steps = info_vi['converge']

fig, ax = plt.subplots(nrows=steps,ncols=2, figsize=(steps * figsize_mul * 0.5, figsize_mul* 3))
for i in range(steps):
    visualize_value_function(ax[i][0],info_vi['v'][i], nx, ny)
    visualize_policy(ax[i][1], info_vi['pi'][i], nx, ny)
