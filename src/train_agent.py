import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train_taxi(episodes=5000, max_steps=200, lr=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
    # Create environment
    env = gym.make("Taxi-v3")

    # Initialize Q-table
    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = np.zeros((state_space, action_space))

    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(action_space)
            else:
                action = np.argmax(q_table[state])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            q_table[state, action] += lr * (
                reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
            )

            total_reward += reward
            state = new_state

            if terminated or truncated:
                break

        # Decay epsilon once per episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{episodes}, avg reward (last 100): {avg_reward:.2f}, epsilon: {epsilon:.3f}")

    env.close()
    return q_table, rewards_per_episode


if __name__ == "__main__":
    q_table, rewards = train_taxi()

    # Plot training curve
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning: Taxi-v3 Training Rewards")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("../media/training_curve.png")
    plt.show()
