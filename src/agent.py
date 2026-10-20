# chess auto agent - lets chess bots play each other on your laptop and measures who is better
#
#   python -m src.agent match minimax:2 random --games 10
#   python -m src.agent match "uci:../chess-learning-bot/build/learnbot uci@200" minimax:2 --games 20
#   python -m src.agent tournament random greedy minimax:1 minimax:2 --games 4
#   python -m src.agent play minimax:2            you vs the bot in the terminal
import argparse
import os
import sys

import chess

from .agents import make_agent
from .match import elo_diff, round_robin, run_match


def close_all(agents):
    for a in agents:
        a["close"]()


def cmd_match(args):
    a, b = make_agent(args.a, seed=1), make_agent(args.b, seed=2)
    try:
        w, d, l = run_match(a, b, args.games, args.max_moves, args.pgn)
    finally:
        close_all([a, b])
    elo, margin = elo_diff(w, d, l)
    n = w + d + l
    print(f"\n{a['name']} vs {b['name']}: +{w} ={d} -{l}  score {(w + d / 2) / n:.0%}")
    if d == 0 and (w == 0 or l == 0):
        # won or lost every game: we only know a lower limit
        print(f"elo difference: {'at least +' if l == 0 else 'at most '}{elo:.0f} (play more games to know more)")
    else:
        print(f"elo difference: {elo:+.0f} (+/- {margin:.0f})")
    if args.pgn:
        print("games saved in", args.pgn)


def cmd_tournament(args):
    agents = [make_agent(s, seed=i) for i, s in enumerate(args.agents)]
    try:
        rows = round_robin(agents, args.games, args.max_moves, args.pgn)
    finally:
        close_all(agents)
    print(f"\n{'#':<3}{'agent':<28}{'points':>8}{'games':>7}{'elo':>8}")
    for i, (name, pts, games, elo) in enumerate(rows, 1):
        print(f"{i:<3}{name:<28}{pts:>8g}{games:>7}{elo:>+8.0f}")
    print(f"\n(elo is compared to {agents[0]['name']} = 0)")


def cmd_play(args):
    bot = make_agent(args.bot)
    board = chess.Board()
    you = chess.WHITE if args.color == "white" else chess.BLACK
    print(f"you are {args.color}. type moves like e4, Nf3 or e2e4. 'quit' to stop")
    try:
        while not board.is_game_over(claim_draw=True):
            print("\n" + str(board) + "\n")
            if board.turn == you:
                text = input("your move: ").strip()
                if text == "quit":
                    return
                try:
                    move = board.parse_san(text)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(text)
                    except ValueError:
                        move = None
                if move not in board.legal_moves:
                    print("not a legal move")
                    continue
            else:
                move = bot["pick"](board)
                print(f"{bot['name']} plays {board.san(move)}")
            board.push(move)
        print("\n" + str(board))
        print("game over:", board.result(claim_draw=True))
    finally:
        bot["close"]()


def main(argv=None):
    p = argparse.ArgumentParser(prog="chess-auto-agent", description="chess bots playing each other, locally")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="two agents play some games")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--games", type=int, default=10)
    m.add_argument("--pgn", help="save games to this pgn file")

    t = sub.add_parser("tournament", help="everyone plays everyone")
    t.add_argument("agents", nargs="+")
    t.add_argument("--games", type=int, default=4, help="games per pair")
    t.add_argument("--pgn")

    pl = sub.add_parser("play", help="you play against an agent")
    pl.add_argument("bot")
    pl.add_argument("--color", default="white", choices=["white", "black"])

    for s in (m, t):
        s.add_argument("--max-moves", type=int, default=200, help="half moves before the game is called a draw")

    args = p.parse_args(argv)
    if getattr(args, "pgn", None):
        os.makedirs(os.path.dirname(os.path.abspath(args.pgn)), exist_ok=True)
    {"match": cmd_match, "tournament": cmd_tournament, "play": cmd_play}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
