###
# Group Members
# 2602515 Taboka Chloe Dube
# 2541693 Wendy Maboa
# 2596852 Liam Brady
# 2333776 Refiloe Mopeloa
###


#to use gym_env: source gym_env/bin/activate


import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


#EXCERCISE 1.1

def generate_random_trajectory(env, max_steps=30):
    state = env.reset()
    trajectory = []
    actions = ["U", "D", "L", "R"] 

    for _ in range(max_steps):
        action = env.action_space.sample() 
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action))
        state = next_state
        if done:
            break
    return trajectory


def print_trajectory_as_grid(env, trajectory):
    grid_size = env.shape[0]
    action_symbols = {0: "U", 1: "R", 2: "D", 3: "L"}
    grid = [["o" for _ in range(grid_size)] for _ in range(grid_size)]

    for (s, a) in trajectory:
        row, col = divmod(s, grid_size)
        grid[row][col] = action_symbols[a]

    for row in grid:
        print(" ".join(row))


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    """
    
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
       
        for s in range(env.observation_space.n):
            v = 0
           
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)



def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Policy Iteration Algorithm.
    Returns optimal policy and value function.
    """
    def one_step_lookahead(state, V):
        """Calculate action values for a given state under value function V."""
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    while True:
        # Policy Evaluation
        V = policy_evaluation_fn(env, policy, discount_factor)

        # Policy Improvement
        policy_stable = True
        for s in range(env.observation_space.n):
            chosen_a = np.argmax(policy[s])
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                policy_stable = False

            # Update policy to greedy action
            policy[s] = np.eye(env.action_space.n)[best_a]

        if policy_stable:
            return policy, V



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    Returns (policy, V).
    """
    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            action_values = one_step_lookahead(s, V)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    # Extract policy
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        action_values = one_step_lookahead(s, V)
        best_a = np.argmax(action_values)
        policy[s, best_a] = 1.0

    return policy, V


def compare_runtimes(env):
    discounts = np.logspace(-0.2, 0, num=30)
    pi_times, vi_times = [], []

    for gamma in discounts:
        # Policy Iteration
        pi_runtime = timeit.timeit(
            lambda: policy_iteration(env, policy_evaluation, discount_factor=gamma),
            number=10
        ) / 10
        pi_times.append(pi_runtime)

        # Value Iteration
        vi_runtime = timeit.timeit(
            lambda: value_iteration(env, discount_factor=gamma),
            number=10
        ) / 10
        vi_times.append(vi_runtime)

    plt.plot(discounts, pi_times, label="Policy Iteration")
    plt.plot(discounts, vi_times, label="Value Iteration")
    plt.xlabel("Discount Factor (γ)")
    plt.ylabel("Average Running Time (s)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.show()



def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("Initial grid:")
    env.render()
    print("")

    # Exercise 1.1: Generate and print random trajectory
    trajectory = generate_random_trajectory(env, max_steps=20)
    print("Random trajectory (actions at visited states):")
    print_trajectory_as_grid(env, trajectory)
    print("")

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")
    
   
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    
    v = policy_evaluation(env, policy)
    print("State values under random policy:")
    print(v.reshape(env.shape))
    print("")

    
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
    print("Policy evaluation test passed!")

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    
    
    policy_pi, v_pi = policy_iteration(env, policy_evaluation)

    print("Optimal state values from policy iteration:")
    print(v_pi.reshape(env.shape))
    print("")

    print("Optimal policy from policy iteration (as actions):")
    action_symbols = {0: "U", 1: "R", 2: "D", 3: "L"}
    policy_grid = []
    for s in range(env.observation_space.n):
        if s in [24]:  
            policy_grid.append("X")
        else:
            policy_grid.append(action_symbols[np.argmax(policy_pi[s])])
    print(np.array(policy_grid).reshape(env.shape))
    print("")

    
    expected_v_pi = np.array([-8., -7., -6., -5., -4.,
                             -7., -6., -5., -4., -3.,
                             -6., -5., -4., -3., -2.,
                             -5., -4., -3., -2., -1.,
                             -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v_pi, expected_v_pi, decimal=1)
    print("Policy iteration test passed!")

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    
    
    policy_vi, v_vi = value_iteration(env)

    print("Optimal state values from value iteration:")
    print(v_vi.reshape(env.shape))
    print("")

    print("Optimal policy from value iteration (as actions):")
    policy_grid_vi = []
    for s in range(env.observation_space.n):
        if s in [24]:  
            policy_grid_vi.append("X")
        else:
            policy_grid_vi.append(action_symbols[np.argmax(policy_vi[s])])
    print(np.array(policy_grid_vi).reshape(env.shape))
    print("")

    
    expected_v_vi = np.array([-8., -7., -6., -5., -4.,
                              -7., -6., -5., -4., -3.,
                              -6., -5., -4., -3., -2.,
                              -5., -4., -3., -2., -1.,
                              -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v_vi, expected_v_vi, decimal=1)
    print("Value iteration test passed!")

    
    compare_runtimes(env)


if __name__ == "__main__":
    main()
