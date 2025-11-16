import chess.pgn
import numpy as np
import torch
from pathlib import Path
from model import encode_board, encode_move
from tqdm import tqdm
import pickle

def parse_pgn_file(pgn_path, max_games=None, min_elo=2200):
    """
    Parse PGN file and extract training data.
    
    Args:
        pgn_path: Path to .pgn file
        max_games: Maximum number of games to parse (None = all)
        min_elo: Minimum ELO rating for both players
    
    Returns:
        List of (board_tensor, move_index, result) tuples
    """
    training_data = []
    games_processed = 0
    positions_extracted = 0
    
    with open(pgn_path) as pgn_file:
        pbar = tqdm(desc="Parsing games")
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            if max_games and games_processed >= max_games:
                break
            
            # Filter by ELO rating
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo < min_elo or black_elo < min_elo:
                    continue
            except (ValueError, TypeError):
                continue
            
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_value = 1.0
            elif result == "0-1":
                game_value = -1.0
            elif result == "1/2-1/2":
                game_value = 0.0
            else:
                continue  # Skip unfinished games
            
            # Extract positions and moves
            board = game.board()
            for move in game.mainline_moves():
                # Encode current position
                board_tensor = encode_board(board, "18")
                
                # Encode the move that was played
                move_index = encode_move(move, board)
                
                # Skip if move encoding failed
                if move_index < 0 or move_index >= 4672:
                    board.push(move)
                    continue
                
                # Value from current player's perspective
                value = game_value if board.turn == chess.WHITE else -game_value
                
                training_data.append({
                    'board': board_tensor.numpy(),
                    'move': move_index,
                    'value': value
                })
                
                positions_extracted += 1
                board.push(move)
            
            games_processed += 1
            pbar.update(1)
            pbar.set_postfix({
                'games': games_processed,
                'positions': positions_extracted
            })
        
        pbar.close()
    
    print(f"\nParsing complete!")
    print(f"Games processed: {games_processed}")
    print(f"Positions extracted: {positions_extracted}")
    
    return training_data


def create_training_dataset(pgn_paths, output_path, max_games_per_file=10000, min_elo=2200):
    """
    Create training dataset from multiple PGN files.
    
    Args:
        pgn_paths: List of paths to PGN files or single path
        output_path: Where to save the dataset (.pkl)
        max_games_per_file: Max games to parse from each file
        min_elo: Minimum ELO rating
    """
    if isinstance(pgn_paths, (str, Path)):
        pgn_paths = [pgn_paths]
    
    all_data = []
    
    for pgn_path in pgn_paths:
        print(f"\nProcessing {pgn_path}...")
        data = parse_pgn_file(pgn_path, max_games=max_games_per_file, min_elo=min_elo)
        all_data.extend(data)
        print(f"Total positions so far: {len(all_data)}")
    
    # Save to file
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Dataset saved! Total positions: {len(all_data)}")
    return all_data


# PyTorch Dataset class
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to .pkl file created by create_training_dataset
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        board_tensor = torch.from_numpy(sample['board']).float()
        
        # Create one-hot policy target
        policy_target = torch.zeros(4672)
        policy_target[sample['move']] = 1.0
        
        value_target = torch.tensor([sample['value']], dtype=torch.float32)
        
        return board_tensor, policy_target, value_target


if __name__ == "__main__":
    from glob import glob
    
    # Get all PGN files from Lichess Elite Database
    pgn_dir = Path("Lichess Elite Database")
    pgn_files = sorted(glob(str(pgn_dir / "*.pgn")))
    
    print(f"Found {len(pgn_files)} PGN files")
    print("Sample files:")
    for f in pgn_files[:5]:
        print(f"  - {Path(f).name}")
    
    # Process a few recent files for good balance of data quality and training speed
    # Using last 5 files (2019-2020) - enough for good training without exponential time
    files_to_process = pgn_files[-5:]  # Last 5 files
    
    print(f"\nProcessing {len(files_to_process)} files...")
    print("Files to process:")
    for f in files_to_process:
        print(f"  - {Path(f).name}")
    
    # Parse and create dataset
    create_training_dataset(
        pgn_paths=files_to_process,
        output_path="chess_training_data.pkl",
        max_games_per_file=5000,  # 5K games per file = ~25K total games, ~150K positions
        min_elo=2200
    )
    
    # Test loading the dataset
    dataset = ChessDataset("chess_training_data.pkl")
    print(f"\nDataset ready with {len(dataset)} positions")
    
    # Show sample
    board, policy, value = dataset[0]
    print(f"Sample board shape: {board.shape}")
    print(f"Sample policy shape: {policy.shape}")
    print(f"Sample value: {value.item()}")
