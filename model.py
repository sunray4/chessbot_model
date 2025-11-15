import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPCN(nn.Module):
    def __init__(self, board_channels=12, policy_size=4672):
        """
        board_channels = 12   -> 6 piece types × 2 colors
        policy_size    = 4672 -> fixed encoding of all legal UCI moves
        """
        super().__init__()

        # === Feature extractor (very tiny CNN) ===
        self.conv1 = nn.Conv2d(board_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # === Policy head ===
        self.policy_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        # === Value head ===
        self.value_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # shared tower
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # value in (-1, 1)

        return policy_logits, value


model = TinyPCN()
for name, param in model.named_parameters():
    print(name, param.data.mean(), param.data.std())

x = torch.randn(1, 12, 8, 8)
policy_logits, value = model(x)

print(policy_logits.shape)  # torch.Size([1, 4672])
print(value)               # value between -1 and 1
