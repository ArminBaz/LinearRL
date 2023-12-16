import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.animation as manimation
import random


def create_transition_matrix_mapping(maze):
        """
        Creates a mapping from maze state indices to transition matrix indices
        """
        n = len(maze)  # Size of the maze (N)

        mapping = {}
        matrix_idx = 0

        for i in range(n):
            for j in range(n):
                mapping[(i,j)] = matrix_idx
                matrix_idx += 1

        return mapping

def get_transition_matrix(env, size, mapping):
        maze = env.unwrapped.maze

        T = np.zeros(shape=(size, size))
        # loop through the maze
        for row in range(maze.shape[0]):
            for col in range(maze.shape[1]):            
                # if we hit a barrier
                if maze[row,col] == '1':
                    continue

                idx_cur = mapping[row, col]

                # check if current state is terminal
                if maze[row,col] == 'G':
                    T[idx_cur, idx_cur] = 1
                    continue

                state = (row,col)
                successor_states = env.unwrapped.get_successor_states(state)
                for successor_state in successor_states:
                    idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                    T[idx_cur, idx_new] = 1/len(successor_states)
        
        return T

def get_map(agent):
    # Replace 'S' and 'G' with 0
    m = np.where(np.isin(agent.maze, ['S', 'G']), '0', agent.maze)

    # Convert the array to int
    m = m.astype(int)
    
    return m

def render_maze(agent, state, ax=None):    
    if ax is None:
        fig, ax = plt.subplots()
    m = get_map(agent)
    
    # Display maze
    ax.imshow(m, origin='upper')
    # Display agent
    agent_loc = patches.Circle((state[1],state[0]), radius=0.4, fill=True, color='white')
    ax.add_patch(agent_loc)
    # Display Reward
    reward = patches.Circle((agent.target_loc[1],agent.target_loc[0]), radius=0.4, fill=True, color='green')
    ax.add_patch(reward)

    ax.set_title('Map')
    ax.set_axis_off()

def render_DR(agent, state, ax=None):
    state_idx = agent.mapping[(state[0], state[1])]
    ax.imshow(agent.DR[state_idx].reshape(agent.height, agent.width), 
              origin='upper', cmap='plasma')
    ax.set_title("DR(%d, %d)" % (state[0], state[1]))
    ax.set_axis_off()

def render_V(agent, ax):
    min_value = np.min(agent.V[~np.isinf(agent.V)])
    max_value = np.max(agent.V)

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)

    ax.imshow(agent.V.reshape(agent.height, agent.width),
                origin='upper',
                cmap=cmap, vmin=min_value, vmax=max_value)
    ax.set_title("$Values$")
    ax.set_axis_off()

def make_plots(agent, state=None):
    # Adjust DR at terminal state
    idx = agent.mapping[agent.target_loc[0], agent.target_loc[1]]
    agent.DR[idx, :] = 0
    agent.DR[idx, idx] = 1

    if state is None:
        state = agent.start_loc
        # state = (0,0)
        
    fig, axs = plt.subplots(1, 3, dpi=144)
    render_maze(agent, state, ax=axs[0])
    render_DR(agent, state, ax=axs[1])
    render_V(agent, ax=axs[2])
    
    plt.show()

def record_trajectory(agent, traj):
    m = get_map(agent)
    
    # Display maze
    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)
    plt.imshow(m, origin='upper', cmap=cmap)
    # Display agent
    agent_loc = patches.Circle((agent.start_loc[1],agent.start_loc[0]), radius=0.4, fill=True, color='blue')
    plt.add_patch(agent_loc)
    # Display Reward
    reward = patches.Circle((agent.target_loc[1],agent.target_loc[0]), radius=0.4, fill=True, color='green')
    plt.add_patch(reward)

    # loop through trajectory and add arrows
    for loc in traj:
        arrow = patches.Arrow((loc[1],loc[0]), fill=True, color='red')
        plt.add_patch(arrow)

    plt.set_title('Map')
    plt.set_axis_off()

def record_trials(agent, title="recorded_trials", n_trial_per_loc=1,
                    start_locs=None, max_steps=100):
    metadata = dict(title=title, artist='JG')
    writer = manimation.FFMpegFileWriter(fps=10, metadata=metadata)
    fig, axs = plt.subplots(1, 3, figsize=(7, 3))
    fig.tight_layout()

    with writer.saving(fig, "./out/%s.mp4" % title, 144):
        for sl in start_locs:
            for trial in range(n_trial_per_loc):
                agent.env.reset()
                done = False
                steps = 0
                state = sl
                
                # set the start and agent location
                agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state
                # Render starting state
                render_maze(agent, state, ax=axs[0])
                render_DR(agent, state, ax=axs[1])
                render_V(agent, ax=axs[2])
                writer.grab_frame()
                for ax in axs:
                        ax.clear()

                # Act greedily and record each state as well
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    obs, _, done, _, _ = agent.env.step(action)

                    render_maze(agent, state, ax=axs[0])
                    render_DR(agent, state, ax=axs[1])
                    render_V(agent, ax=axs[2])
                    writer.grab_frame()

                    steps += 1

                    state = obs["agent"]

                    for ax in axs:
                        ax.clear()

def record_trajectory(agent, traj):
    fig, ax = plt.subplots()

    m = get_map(agent)
    
    # Display maze
    cmap = plt.cm.binary
    # cmap.set_bad('black', 1.0)
    ax.imshow(m, origin='upper', cmap=cmap)
    # Display agent
    agent_loc = patches.Circle((agent.start_loc[1],agent.start_loc[0]), alpha=0.7, radius=0.4, fill=True, color='blue')
    ax.add_patch(agent_loc)
    # Display Reward
    reward = patches.Circle((agent.target_loc[1],agent.target_loc[0]), alpha=0.7, radius=0.4, fill=True, color='green')
    ax.add_patch(reward)

    # loop through trajectory and add arrows
    for i in range(0,len(traj)-1):
        diff = traj[i+1] - traj[i]
        arrow = patches.FancyArrow(x=traj[i][1], y=traj[i][0], dx=0.5*diff[1], dy=0.5*diff[0], width=0.06, length_includes_head=True, color="red")
        # arrow = patches.Arrow((loc[1],loc[0]), fill=True, color='red')
        ax.add_patch(arrow)

    # ax.set_title('Map')
    ax.set_axis_off()

def test_agent(agent, state=None):
    """
    Function to test the agent
    """
    traj = []

    agent.env.reset()
    if state is None:
        state = agent.start_loc

    # set the start and agent location
    agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state
    print(f"Starting in state: {state}")
    steps = 0
    done = False
    while not done:
        action = agent.select_action(state)
        obs, _, done, _, _ = agent.env.step(action)
        next_state = obs["agent"]
        traj.append(next_state)
        print(f"Took action: {action} and arrived in state: {next_state}")

        steps += 1
        state = next_state
    print(f"Took {steps} steps")

    return traj