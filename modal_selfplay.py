import modal
import chess
import numpy as np
import torch

# Create Modal app
app = modal.App("chess-selfplay")

# Define image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "chess")
)

# Mount your code files
mounts = [
    modal.Mount.from_local_file("model.py", remote_path="/root/model.py"),
    modal.Mount.from_local_file("mcts.py", remote_path="/root/mcts.py"),
]

@app.function(
    image=image,
    mounts=mounts,
    timeout=3600,  # 1 hour timeout
    cpu=2.0,
)
def run_selfplay_games(num_games=10, num_simulations=20):
    """Run self-play games and return the data."""
    import sys
    sys.path.insert(0, '/root')
    
    from mcts import MCTSNode, mcts_search
    from model import TinyPCN, encode_board, encode_move
    
    net = TinyPCN()
    all_games = []
    
    for game_idx in range(num_games):
        board = chess.Board()
        game_data = []
        move_count = 0
        print(f"Starting game {game_idx+1}/{num_games}...")
        
        while not board.is_game_over():
            root = MCTSNode(board)
            move_visits = mcts_search(root, net, num_simulations=num_simulations)
            policy = np.zeros(4672)
            
            for move, visits in move_visits.items():
                idx = encode_move(move, board)
                if idx >= 0 and idx < 4672:
                    policy[idx] = visits
            
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
            else:
                policy = np.ones(4672) / 4672
            
            game_data.append((encode_board(board, "18").numpy(), policy, board.turn))
            move = max(move_visits, key=move_visits.get)
            board.push(move)
            move_count += 1
            
            if move_count % 10 == 0:
                print(f"  Move {move_count}...")
        
        result = board.result()
        value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result]
        
        for i in range(len(game_data)):
            board_tensor, policy, turn = game_data[i]
            game_data[i] = (board_tensor, policy, value if turn else -value)
        
        all_games.append(game_data)
        print(f"Game {game_idx+1}/{num_games} finished: {result}")
    
    return all_games


@app.local_entrypoint()
def main():
    """Entry point to run from local machine."""
    print("Starting self-play on Modal...")
    games_data = run_selfplay_games.remote(num_games=10, num_simulations=20)
    print(f"\nCompleted! Generated {len(games_data)} games.")
    
    # Optionally save the data
    import pickle
    with open("selfplay_data_modal.pkl", "wb") as f:
        pickle.dump(games_data, f)
    print("Data saved to selfplay_data_modal.pkl")
