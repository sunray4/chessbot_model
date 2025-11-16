import chess
import numpy as np
import torch
from mcts import MCTSNode, mcts_search
from model import TinyPCN, encode_board, encode_move

# Configuration
NUM_GAMES = 10  # Set to desired number of games
NUM_SIMULATIONS = 20  # MCTS simulations per move (reduced for speed)

net = TinyPCN()
all_games = []

for game_idx in range(NUM_GAMES):
    board = chess.Board()
    game_data = []
    move_count = 0
    print(f"Starting game {game_idx+1}/{NUM_GAMES}...")
    while not board.is_game_over():
        root = MCTSNode(board)
        move_visits = mcts_search(root, net, num_simulations=NUM_SIMULATIONS)
        policy = np.zeros(4672)
        for move, visits in move_visits.items():
            idx = encode_move(move, board)
            if idx >= 0 and idx < 4672:
                policy[idx] = visits
        if np.sum(policy) > 0:
            policy = policy / np.sum(policy)
        else:
            policy = np.ones(4672) / 4672  # fallback uniform
        game_data.append((encode_board(board, "18"), policy, board.turn))
        move = max(move_visits, key=move_visits.get)
        board.push(move)
        move_count += 1
        if move_count % 5 == 0:
            print(f"  Move {move_count}...")
    result = board.result()
    value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result]
    for i in range(len(game_data)):
        game_data[i] = (game_data[i][0], game_data[i][1], value if game_data[i][2] else -value)
    all_games.append(game_data)
    print(f"Game {game_idx+1}/{NUM_GAMES} finished: {result}")

# Optionally: save all_games to disk for later training
# Example: torch.save(all_games, 'selfplay_data.pt')
