
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Select GPU automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Network Architecture
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()

        self.input_size = input_size
        self.nb_action = nb_action

        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    # Forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Experience Replay
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)

        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):

        samples = random.sample(self.memory, batch_size)

        batch_state, batch_next_state, batch_action, batch_reward = zip(*samples)

        return (
            torch.cat(batch_state, dim=0),
            torch.cat(batch_next_state, dim=0),
            torch.cat(batch_action, dim=0),
            torch.cat(batch_reward, dim=0)
        )


# Deep Q Network
class Dqn:

    def __init__(self, input_size, nb_action, gamma):

        self.gamma = gamma
        self.reward_window = []

        self.device = DEVICE

        print("DQN running on:", self.device)

        self.model = Network(input_size, nb_action).to(self.device)

        self.memory = ReplayMemory(10000)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.last_state = torch.zeros(
            1, input_size, dtype=torch.float32
        )

        self.last_action = 0
        self.last_reward = 0.0

    # Action selection
    def select_action(self, state):

        state = state.to(self.device)

        with torch.no_grad():

            q_values = self.model(state)

            probs = F.softmax(q_values * 100, dim=1)

            action = torch.multinomial(probs, 1)

        return action.item()

    # Learning
    def learn(self, batch_state, batch_next_state,
              batch_action, batch_reward):

        batch_state = batch_state.to(self.device)
        batch_next_state = batch_next_state.to(self.device)
        batch_action = batch_action.to(self.device)
        batch_reward = batch_reward.to(self.device)

        # Current Q values
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)
        ).squeeze(1)

        # Next state Q values
        with torch.no_grad():
            next_outputs = self.model(batch_next_state).max(1)[0]

            target = batch_reward + self.gamma * next_outputs

        # TD Loss
        td_loss = F.smooth_l1_loss(outputs, target)

        self.optimizer.zero_grad()

        td_loss.backward()

        self.optimizer.step()

    # Update experience
    def update(self, reward, new_signal):

        new_state = torch.tensor(
            new_signal, dtype=torch.float32
        ).unsqueeze(0)

        self.memory.push((
            self.last_state,
            new_state,
            torch.tensor([self.last_action], dtype=torch.long),
            torch.tensor([self.last_reward], dtype=torch.float32)
        ))

        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:

            batch_state, batch_next_state, batch_action, batch_reward = (
                self.memory.sample(100)
            )

            self.learn(
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward
            )

        self.last_action = action
        self.last_state = new_state
        self.last_reward = float(reward)

        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    # Calculate average score
    def score(self):

        if not self.reward_window:
            return 0.0

        return sum(self.reward_window) / len(self.reward_window)

    # Save model
    def save(self):

        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, "last_brain.pth")

        print("Model saved successfully!")

    # Load model
    def load(self):

        if os.path.isfile("last_brain.pth"):

            print("Loading checkpoint...")

            checkpoint = torch.load(
                "last_brain.pth",
                map_location=self.device
            )

            self.model.load_state_dict(checkpoint["state_dict"])

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            print("Model loaded successfully!")

        else:
            print("No checkpoint found!")
