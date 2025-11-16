import torch
import chess
from model import TinyPCN, encode_board

def test_forward_shapes():
    net = TinyPCN()
    x = torch.randn(2, 12, 8, 8)
    policy, value = net(x)
    assert policy.shape == (2, 4672)
    assert value.shape == (2, 1)

def test_encode_board_planes():
    board = chess.Board()
    planes = encode_board(board, "20")
    assert planes.shape == (20, 8, 8)