import numpy as np
import torch
import chess
from model import TinyPCN, encode_board, encode_move

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value
        self.P = None  # prior probability

    def is_expanded(self):
        return len(self.children) > 0


def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_child = None
    for move, child in node.children.items():
        u = c_puct * child.P * np.sqrt(node.N) / (1 + child.N)
        score = child.Q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def expand_node(node, net):
    board_tensor = encode_board(node.board, "18").unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = net(board_tensor)
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        value = float(value.item())
    legal_moves = list(node.board.legal_moves)
    move_indices = [encode_move(m, node.board) for m in legal_moves]
    # Filter out invalid indices and use safe indexing
    policy_scores = []
    for i in move_indices:
        if i >= 0 and i < len(policy):
            policy_scores.append(policy[i])
        else:
            policy_scores.append(1e-9)  # small prior for invalid moves
    priors = softmax(policy_scores)
    node.P = 1.0  # root prior
    for move, p in zip(legal_moves, priors):
        next_board = node.board.copy()
        next_board.push(move)
        node.children[move] = MCTSNode(next_board, parent=node, move=move)
        node.children[move].P = p
    return value


def backup(node, value):
    while node:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value  # switch perspective
        node = node.parent


def mcts_search(root, net, num_simulations=100, c_puct=1.0):
    for _ in range(num_simulations):
        node = root
        # Selection
        while node.is_expanded() and node.children:
            node = select_child(node, c_puct)
        # Expansion & Evaluation
        value = expand_node(node, net)
        # Backup
        backup(node, value)
    # Return visit counts for root's children
    move_visits = {move: child.N for move, child in root.children.items()}
    return move_visits

# Example usage:
if __name__ == "__main__":
    net = TinyPCN()
    board = chess.Board()
    root = MCTSNode(board)
    mcts_search(root, net, num_simulations=50)
    print({str(move): n for move, n in root.children.items()})
