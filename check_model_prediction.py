"""
Quick script to check what your model predicts for a chess position.
Shows if the model thinks it's winning, drawing, or losing.
"""

import torch
import chess
from model import TinyPCN, encode_board

def check_position(board_fen=None, model_path="chess_model.pth"):
    """
    Check what the model predicts for a given position.
    
    Args:
        board_fen: FEN string of position (None = starting position)
        model_path: Path to trained model
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyPCN(board_channels=18, policy_size=4672)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Set up position
    if board_fen is None:
        board = chess.Board()  # Starting position
        print("Position: Starting position")
    else:
        board = chess.Board(board_fen)
        print(f"Position: {board_fen}")
    
    print(board)
    print()
    
    # Get model prediction
    board_tensor = encode_board(board).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    value_score = value.item()
    
    # Interpret the value
    print(f"Model's value prediction: {value_score:.4f}")
    print()
    print("Interpretation:")
    
    if value_score > 0.7:
        print("  🟢 STRONG ADVANTAGE - Position is clearly winning")
    elif value_score > 0.3:
        print("  🟢 ADVANTAGE - Position is favorable")
    elif value_score > 0.1:
        print("  🟡 SLIGHT ADVANTAGE - Small edge")
    elif value_score > -0.1:
        print("  ⚪ EQUAL/DRAW - Position is balanced")
    elif value_score > -0.3:
        print("  🟡 SLIGHT DISADVANTAGE - Small edge for opponent")
    elif value_score > -0.7:
        print("  🔴 DISADVANTAGE - Position is unfavorable")
    else:
        print("  🔴 STRONG DISADVANTAGE - Position is clearly losing")
    
    print()
    print(f"From {('White' if board.turn else 'Black')}'s perspective:")
    if board.turn == chess.WHITE:
        if value_score > 0.1:
            print(f"  White is better by {value_score:.2f}")
        elif value_score < -0.1:
            print(f"  Black is better by {-value_score:.2f}")
        else:
            print(f"  Position is roughly equal")
    else:
        if value_score > 0.1:
            print(f"  Black is better by {value_score:.2f}")
        elif value_score < -0.1:
            print(f"  White is better by {-value_score:.2f}")
        else:
            print(f"  Position is roughly equal")
    
    # Show top moves too
    policy_probs = torch.softmax(policy_logits, dim=1)[0]
    print()
    print("Top 5 moves the model would consider:")
    
    from model import decode_move, encode_move
    legal_moves = list(board.legal_moves)
    legal_indices = []
    legal_move_map = {}
    
    for move in legal_moves:
        idx = encode_move(move, board)
        if idx >= 0 and idx < 4672:
            legal_indices.append(idx)
            legal_move_map[idx] = move
    
    if legal_indices:
        legal_probs = policy_probs[legal_indices]
        top_k = min(5, len(legal_indices))
        top_indices = torch.topk(legal_probs, top_k).indices
        
        for i, idx in enumerate(top_indices):
            move_idx = legal_indices[idx]
            move = legal_move_map[move_idx]
            prob = legal_probs[idx].item()
            print(f"  {i+1}. {move.uci()} ({prob*100:.2f}%)")
    
    return value_score


if __name__ == "__main__":
    print("="*60)
    print("CHESS MODEL POSITION EVALUATOR")
    print("="*60)
    print()
    
    # Test 1: Starting position
    print("TEST 1: Starting position (should be ~0.0 for equal)")
    print("-" * 60)
    check_position()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: White is clearly winning (Queen vs pawn endgame)
    print("TEST 2: White winning position")
    print("-" * 60)
    fen_white_winning = "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"
    check_position(fen_white_winning)
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Black is clearly winning
    print("TEST 3: Black winning position")
    print("-" * 60)
    fen_black_winning = "4k3/4q3/8/8/8/8/8/4K3 w - - 0 1"
    check_position(fen_black_winning)
    
    print("\n" + "="*60 + "\n")
    
    # Test 4: Scholar's mate position (White about to checkmate)
    print("TEST 4: White delivers checkmate next move")
    print("-" * 60)
    fen_mate_in_1 = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
    check_position(fen_mate_in_1)
    
    print("\n" + "="*60)
    print("\nDone! You can also call this script with custom FEN strings:")
    print('  python check_model_prediction.py')
