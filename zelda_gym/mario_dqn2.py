import numpy as np
from mario_bizhawk_env2 import MarioBizHawkEnv2
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[key])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            key = self.get_state_key(state)
            next_key = self.get_state_key(next_state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_size)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.action_size)
            target = reward
            if not done:
                target += self.gamma * np.amax(self.q_table[next_key])
            self.q_table[key][action] += self.learning_rate * (target - self.q_table[key][action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env = MarioBizHawkEnv2()
    state_size = env.state_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    episodes = 100
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.replay()
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    main()
