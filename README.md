# Chess bot model

A chess playing neural network trained on expert games from the Lichess Elite Database.

## Model description

This is a policy-value neural network inspired by AlphaZero, designed to evaluate chess positions and suggest moves.

## Architecture
- Input: 18-plane board representation (12 pieces + 6 metadata planes)
- Convolutional backbone: 32 filters, 1 residual block, ~9,611,202 parameters
- Policy head: 4,672-dimensional output (one per legal move encoding)
- Value head: Single tanh output (-1 to +1 for position evaluation)

## Training data
- Dataset: Lichess Elite Database (games from 2200+ ELO players)
- Positions trained: 16,800,000
- Epochs: 10

## Performance
- Final Policy Loss: 2.8000
- Final Value Loss: 0.8500

## Usage
```python
import torch
import chess
from huggingface_hub import hf_hub_download
import importlib.util

# Download model files from HuggingFace
model_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="chess_model.pth")
model_py_path = hf_hub_download(repo_id="AubreeL/chess-bot", filename="model.py")

# Import the model architecture from the downloaded file
spec = importlib.util.spec_from_file_location("chess_model", model_py_path)
if spec is None or spec.loader is None:
    raise ImportError("Could not load model from HuggingFace")
chess_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chess_model_module)

TinyPCN = chess_model_module.TinyPCN
encode_board = chess_model_module.encode_board
encode_move = chess_model_module.encode_move

# Initialize and load the model
model = TinyPCN(board_channels=18, policy_size=4672)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Evaluate a position
board = chess.Board()
board_tensor = encode_board(board).unsqueeze(0)

with torch.no_grad():
    policy_logits, value = model(board_tensor)

# Value interpretation:
# +1.0 = winning for current player
#  0.0 = drawn/equal position
# -1.0 = losing for current player

print(f"Position evaluation: {value.item():.4f}")
```