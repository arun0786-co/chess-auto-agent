# the players. every agent is a small dict:  {"name": ..., "pick": fn(board) -> move, "close": fn()}
#
#   random            any legal move
#   greedy            takes the biggest piece it can, else random
#   minimax:3         alpha beta search 3 moves deep (my own eval)
#   uci:<command>     any uci engine, like stockfish or my learnbot.  add @300 for 300 ms per move,
#                     and =name to give it a name (uci:stockfish@300=fish)
import random
import shlex

import chess
import chess.engine

VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# simple piece square tables (white's view, a1 first). pawns like to go forward, knights like the centre
PST = {
    chess.PAWN: [0] * 8 + [5, 10, 10, -20, -20, 10, 10, 5] + [5, -5, -10, 0, 0, -10, -5, 5] + [0, 0, 0, 20, 20, 0, 0, 0]
    + [5, 5, 10, 25, 25, 10, 5, 5] + [10, 10, 20, 30, 30, 20, 10, 10] + [50] * 8 + [0] * 8,
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 5, 10, 15, 15, 10, 5, -30,
                   -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 10, 15, 15, 10, 0, -30,
                   -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10, 10, -10,
                   -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 5, 5, 0, 0, 0] + [-5, 0, 0, 0, 0, 0, 0, -5] * 5 + [5, 10, 10, 10, 10, 10, 10, 5] + [0] * 8,
    chess.QUEEN: [0] * 64,
    chess.KING: [20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20] + [-10] * 8 + [-20] * 8 + [-30] * 32,
}

MATE = 100000


def evaluate(board):
    """score in centipawns for the side to move"""
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    s = 0
    for sq, pc in board.piece_map().items():
        idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
        v = VALUE[pc.piece_type] + PST[pc.piece_type][idx]
        s += v if pc.color == chess.WHITE else -v
    return s if board.turn == chess.WHITE else -s


def order(board, moves):
    # captures of big pieces by small pieces first, then promotions, then the rest
    def key(m):
        k = 0
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            k += 10 * VALUE[victim.piece_type if victim else chess.PAWN] - VALUE[attacker.piece_type]
        if m.promotion:
            k += 800
        return -k
    return sorted(moves, key=key)


def quiesce(board, alpha, beta, depth=6):
    stand = evaluate(board)
    if depth == 0 or stand >= beta:
        return stand
    alpha = max(alpha, stand)
    for m in order(board, [m for m in board.legal_moves if board.is_capture(m)]):
        board.push(m)
        v = -quiesce(board, -beta, -alpha, depth - 1)
        board.pop()
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return alpha


def alphabeta(board, depth, alpha, beta, ply=0):
    if board.is_game_over() or board.can_claim_threefold_repetition():
        if board.is_checkmate():
            return -MATE + ply  # sooner mate is better
        return 0
    if depth == 0:
        return quiesce(board, alpha, beta)
    best = -MATE * 2
    for m in order(board, list(board.legal_moves)):
        board.push(m)
        v = -alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        best = max(best, v)
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


def minimax_move(board, depth, rng=None):
    best, best_moves = -MATE * 2, []
    alpha, beta = -MATE * 2, MATE * 2
    for m in order(board, list(board.legal_moves)):
        board.push(m)
        # window is 1 lower than the best so far: a move that only ties gets its real score,
        # a worse move gets a bound below it (else a bound can look like a tie)
        v = -alphabeta(board, depth - 1, -beta, -(alpha - 1), 1)
        board.pop()
        if v > best:
            best, best_moves = v, [m]
            alpha = max(alpha, v)
        elif v == best:
            best_moves.append(m)
    return (rng or random).choice(best_moves)  # same score -> pick one, so games are not all the same


def random_agent(seed=None):
    rng = random.Random(seed)
    return {"name": "random", "pick": lambda b: rng.choice(list(b.legal_moves)), "close": lambda: None}


def greedy_agent(seed=None):
    rng = random.Random(seed)

    def pick(b):
        caps = [m for m in b.legal_moves if b.is_capture(m)]
        if not caps:
            return rng.choice(list(b.legal_moves))
        def gain(m):
            victim = b.piece_at(m.to_square)
            return VALUE[victim.piece_type] if victim else VALUE[chess.PAWN]  # en passant
        top = max(gain(m) for m in caps)
        return rng.choice([m for m in caps if gain(m) == top])
    return {"name": "greedy", "pick": pick, "close": lambda: None}


def minimax_agent(depth=2, seed=None):
    rng = random.Random(seed)
    return {"name": f"minimax:{depth}", "pick": lambda b: minimax_move(b, depth, rng), "close": lambda: None}


def uci_agent(command, ms=200, name=None):
    eng = chess.engine.SimpleEngine.popen_uci(shlex.split(command))
    limit = chess.engine.Limit(time=ms / 1000)
    label = name or eng.id.get("name", command.split()[0])

    def pick(b):
        return eng.play(b, limit).move

    def close():
        try:
            eng.quit()
        except Exception:
            pass
    return {"name": label, "pick": pick, "close": close}


def make_agent(spec, seed=None):
    """'random', 'greedy', 'minimax:3', 'uci:stockfish@300', 'uci:../learnbot uci@100=learnbot'"""
    if spec == "random":
        return random_agent(seed)
    if spec == "greedy":
        return greedy_agent(seed)
    if spec.startswith("minimax"):
        depth = int(spec.split(":")[1]) if ":" in spec else 2
        return minimax_agent(depth, seed)
    if spec.startswith("uci:"):
        cmd, _, rest = spec[4:].partition("@")
        ms, _, label = rest.partition("=")  # "@200=my name" gives it a name
        return uci_agent(cmd, int(ms) if ms else 200, name=label or None)
    raise ValueError(f"dont know the agent '{spec}'. try random, greedy, minimax:2 or uci:<command>")
