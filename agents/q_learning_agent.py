import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, board):
        return str(board)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [self.q_table.get((state, a), 0) for a in actions]
        return actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, next_actions):
        old_q = self.q_table.get((state, action), 0)

        future_q = 0
        if next_actions:
            future_q = max([self.q_table.get((next_state, a), 0) for a in next_actions])

        new_q = old_q + self.lr * (reward + self.gamma * future_q - old_q)
        self.q_table[(state, action)] = new_q

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="q_table.pkl"):
        try:
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
        except:
            pass
