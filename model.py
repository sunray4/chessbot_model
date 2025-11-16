import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

MOVES_PER_SQUARE = 73
POLICY_SIZE = 64 * MOVES_PER_SQUARE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return F.relu(out)


class TinyPCN(nn.Module):
    def __init__(self, board_channels: int = 18, policy_size: int = POLICY_SIZE) -> None:
        """Tiny policy-value net: shared trunk plus separate heads."""
        super().__init__()

        self.conv1 = nn.Conv2d(board_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.res_block = ResidualBlock(32)

        self.policy_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        self.value_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res_block(x)

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def board_to_18_planes(board: chess.Board) -> torch.FloatTensor:
    """Return 18 AlphaZero-style planes for the given board."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = (piece.piece_type - 1) + color_offset
        planes[plane_idx, row, col] = 1.0

    planes[12, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        planes[17, ep_row, ep_col] = 1.0

    return torch.from_numpy(planes)


def board_to_20_planes(board: chess.Board) -> torch.FloatTensor:
    """Return 20 planes (18 standard plus repetition and move count)."""
    planes18 = board_to_18_planes(board).numpy()
    extra = np.zeros((2, 8, 8), dtype=np.float32)

    try:
        repetition = board.is_repetition()
    except Exception:
        repetition = False
    extra[0, :, :] = 1.0 if repetition else 0.0

    move_norm = min(board.fullmove_number / 100.0, 1.0)
    extra[1, :, :] = float(move_norm)

    planes20 = np.concatenate([planes18, extra], axis=0)
    return torch.from_numpy(planes20)


def encode_board(board: chess.Board, variant: str = "18") -> torch.FloatTensor:
    if variant == "18":
        return board_to_18_planes(board)
    if variant == "20":
        return board_to_20_planes(board)
    raise ValueError("variant must be '18' or '20'")


_RAY_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT_OFFSETS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_PROMOTION_OFFSETS = ((1, 0), (1, 1), (1, -1), (2, 0))
_PROMOTION_PIECES = ("q", "r", "b", "n")

_move_to_index: dict[tuple[int, int, str | None], int] = {}
_index_to_move: dict[int, tuple[int, int, str | None]] = {}


def _init_move_tables() -> None:
    idx = 0
    for sq in range(64):
        row0, col0 = divmod(sq, 8)

        for dx, dy in _RAY_OFFSETS:
            for step in range(1, 8):
                row = row0 + dx * step
                col = col0 + dy * step
                if 0 <= row < 8 and 0 <= col < 8:
                    target = row * 8 + col
                    _move_to_index[(sq, target, None)] = idx
                    _index_to_move[idx] = (sq, target, None)
                idx += 1

        for dx, dy in _KNIGHT_OFFSETS:
            row = row0 + dx
            col = col0 + dy
            if 0 <= row < 8 and 0 <= col < 8:
                target = row * 8 + col
                _move_to_index[(sq, target, None)] = idx
                _index_to_move[idx] = (sq, target, None)
            idx += 1

        for dx, dy in _PROMOTION_OFFSETS:
            row = row0 + dx
            col = col0 + dy
            if 0 <= row < 8 and 0 <= col < 8:
                target = row * 8 + col
                for promo in _PROMOTION_PIECES:
                    _move_to_index[(sq, target, promo)] = idx
                    _index_to_move[idx] = (sq, target, promo)
                    idx += 1
            else:
                idx += len(_PROMOTION_PIECES)

        while idx % MOVES_PER_SQUARE != 0:
            _index_to_move[idx] = None
            idx += 1


_init_move_tables()


def _promotion_symbol(piece_type: int | None) -> str | None:
    if piece_type is None:
        return None
    return chess.Piece(piece_type, chess.WHITE).symbol().lower()


def encode_move(move: chess.Move, board: chess.Board) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo_symbol = _promotion_symbol(move.promotion)

    if board.color_at(move.from_square) == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    key = (from_sq, to_sq, promo_symbol)
    return _move_to_index.get(key, -1)


def decode_move(index: int, board: chess.Board | None = None) -> chess.Move | None:
    triple = _index_to_move.get(index)
    if triple is None:
        return None

    from_sq, to_sq, promo = triple

    if board is not None and board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    promotion = chess.Piece.from_symbol(promo.upper()).piece_type if promo else None
    return chess.Move(from_sq, to_sq, promotion=promotion)
