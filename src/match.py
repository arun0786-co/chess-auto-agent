# playing games between agents, saving them as pgn, and turning results into elo
import math
from datetime import date

import chess
import chess.pgn

# short openings so the games dont all start the same. each pair is played with both colours
OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["e2e4", "c7c5"], ["c2c4", "e7e5"], ["e2e4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"], ["g1f3", "d7d5"], ["e2e4", "c7c6"], ["d2d4", "f7f5"],
]


def play_game(white, black, opening=(), max_moves=200):
    """returns a pgn game. max_moves counts both sides, after that its a draw"""
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_moves:
        side = white if board.turn == chess.WHITE else black
        move = side["pick"](board)
        if move not in board.legal_moves:
            raise RuntimeError(f"{side['name']} played an illegal move {move}")
        board.push(move)
    game = chess.pgn.Game.from_board(board)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # stopped because of the move limit
        game.headers["Termination"] = "move limit"
    game.headers.update({"Event": "chess-auto-agent", "Date": date.today().strftime("%Y.%m.%d"),
                         "White": white["name"], "Black": black["name"], "Result": result})
    return game


def score_of(result, as_white):
    s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
    return s if as_white else 1 - s


def run_match(a, b, games, max_moves=200, pgn_path=None, show=print):
    """a vs b, swapping colours every game. returns (wins, draws, losses) for a"""
    w = d = l = 0
    out = open(pgn_path, "a") if pgn_path else None
    try:
        for g in range(games):
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            a_white = g % 2 == 0
            game = play_game(a, b, opening, max_moves) if a_white else play_game(b, a, opening, max_moves)
            s = score_of(game.headers["Result"], a_white)
            w, d, l = w + (s == 1), d + (s == 0.5), l + (s == 0)
            if show:
                show(f"game {g + 1}: {game.headers['White']} vs {game.headers['Black']}  {game.headers['Result']}"
                     f"   ({a['name']} {w + d / 2:g}/{g + 1})")
            if out:
                print(game, file=out, end="\n\n")
    finally:
        if out:
            out.close()
    return w, d, l


def elo_diff(w, d, l):
    """elo difference with a 95% error margin (normal approximation)"""
    n = w + d + l
    if n == 0:
        return 0.0, 0.0
    s = (w + d / 2) / n
    s_c = min(max(s, 0.5 / n), 1 - 0.5 / n)  # avoid infinity at 0% or 100%
    elo = -400 * math.log10(1 / s_c - 1)
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / n
    se = math.sqrt(var / n)
    lo, hi = min(max(s - 1.96 * se, 1e-3), 1 - 1e-3), min(max(s + 1.96 * se, 1e-3), 1 - 1e-3)
    margin = (-400 * math.log10(1 / hi - 1) - -400 * math.log10(1 / lo - 1)) / 2
    return elo, margin


def round_robin(agents, games, max_moves=200, pgn_path=None, show=print):
    """everyone plays everyone. returns rows sorted by points: (name, points, games, elo)"""
    pts = {a["name"]: 0.0 for a in agents}
    played = {a["name"]: 0 for a in agents}
    pair_scores = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            w, d, l = run_match(a, b, games, max_moves, pgn_path, show=None)
            pts[a["name"]] += w + d / 2
            pts[b["name"]] += l + d / 2
            played[a["name"]] += games
            played[b["name"]] += games
            pair_scores.append((a["name"], b["name"], w + d / 2, games))
            if show:
                show(f"{a['name']} vs {b['name']}: +{w} ={d} -{l}")
    ratings = fit_ratings(list(pts), pair_scores)
    rows = [(n, pts[n], played[n], ratings[n]) for n in pts]
    return sorted(rows, key=lambda r: -r[1])


def fit_ratings(names, pair_scores, steps=2000):
    """simple elo fit: move ratings until expected scores match the real ones. first player stays at 0"""
    r = {n: 0.0 for n in names}
    for _ in range(steps):
        grad = {n: 0.0 for n in names}
        for a, b, s, n in pair_scores:
            exp = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            s = min(max(s, 0.25), n - 0.25)  # a 100% score would push ratings to infinity
            grad[a] += s - n * exp
            grad[b] -= s - n * exp
        for n in names:
            r[n] += 4 * grad[n]
        base = r[names[0]]
        for n in names:
            r[n] -= base
    return r
