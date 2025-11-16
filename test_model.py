import torch
import chess

from model import TinyPCN, encode_board, encode_move, decode_move


def test_tinypcn_forward_shapes():
    net = TinyPCN()
    x = torch.randn(2, 12, 8, 8)
    policy, value = net(x)
    assert policy.shape == (2, 4672)
    assert value.shape == (2, 1)
    assert torch.all(value <= 1.0) and torch.all(value >= -1.0)


def test_encode_board_18_planes_shape():
    board = chess.Board()
    planes = encode_board(board, "18")
    assert planes.shape == (18, 8, 8)
    assert planes.dtype == torch.float32


def test_encode_board_20_planes_shape():
    board = chess.Board()
    planes = encode_board(board, "20")
    assert planes.shape == (20, 8, 8)
    assert planes.dtype == torch.float32


def test_move_encode_decode_roundtrip():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = encode_move(move, board)
    assert idx >= 0
    decoded = decode_move(idx)
    assert decoded is not None
    assert decoded.from_square == move.from_square
    assert decoded.to_square == move.to_square
    assert decoded.promotion == move.promotion
