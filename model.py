import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor):
            x = x.float()
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = model.device

    def _convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            tensor = data.clone().detach()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.FloatTensor(data)
        return tensor.float().to(self.device)

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors using the helper method
        state = self._convert_to_tensor(state)
        next_state = self._convert_to_tensor(next_state)
        action = self._convert_to_tensor(action)
        reward = self._convert_to_tensor(reward)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Get predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_state_value = self.model(next_state[idx].unsqueeze(0))
                Q_new = reward[idx] + self.gamma * torch.max(next_state_value)
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
