
import torch
from agent import Dqn

agent = Dqn(
    input_size=5,
    nb_action=3,
    gamma=0.9
)

# Test state
state = [0.1, 0.2, 0.3, 0.4, -0.4]

# Test action selection and learning updates
for i in range(150):

    reward = 0.1

    action = agent.update(reward, state)

    if i % 25 == 0:
        print("Step:", i, "Action:", action)

print("Device:", agent.device)
print("GPU:", torch.cuda.get_device_name(0))
print("DQN test completed!")
