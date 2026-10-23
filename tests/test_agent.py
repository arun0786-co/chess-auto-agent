import os
import shutil

import chess
import pytest

from src.agent import main
from src.agents import evaluate, make_agent, minimax_move
from src.match import elo_diff, fit_ratings, play_game, run_match, score_of

LEARNBOT = os.path.join(os.path.dirname(__file__), "..", "..", "chess-learning-bot", "build", "learnbot")


def test_eval_start_is_equal():
    assert evaluate(chess.Board()) == 0


def test_eval_sees_extra_queen():
    b = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(b) > 800
    b.turn = chess.BLACK
    assert evaluate(b) < -800


def test_minimax_mates_in_one():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert minimax_move(b, 2).uci() == "a1a8"


def test_minimax_takes_free_queen():
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert minimax_move(b, 2).uci() == "d1d5"


def test_minimax_does_not_hang_queen():
    # Qxb7 grabs a pawn but the rook takes the queen back
    b = chess.Board("1r2k3/1p6/8/8/8/8/8/1Q2K3 w - - 0 1")
    assert minimax_move(b, 2).uci() != "b1b7"


def test_game_is_legal_and_finishes():
    g = play_game(make_agent("random", 1), make_agent("random", 2), max_moves=60)
    assert g.headers["Result"] in ("1-0", "0-1", "1/2-1/2")
    assert len(list(g.mainline_moves())) <= 60


def test_minimax_beats_random():
    w, d, l = run_match(make_agent("minimax:1", 1), make_agent("random", 2), 4, show=None)
    assert w + d / 2 >= 3


def test_scores():
    assert score_of("1-0", True) == 1 and score_of("1-0", False) == 0
    assert score_of("1/2-1/2", False) == 0.5


def test_elo():
    assert elo_diff(5, 0, 5)[0] == pytest.approx(0)
    elo, margin = elo_diff(30, 10, 10)
    assert 140 < elo < 155 and margin > 0  # 70% score is about +147


def test_fit_ratings_order():
    r = fit_ratings(["a", "b", "c"], [("a", "b", 2, 10), ("a", "c", 1, 10), ("b", "c", 3, 10)])
    assert r["a"] == 0 and r["c"] > r["b"] > r["a"]


def test_bad_agent_name():
    with pytest.raises(ValueError):
        make_agent("superbot")


def test_cli_match_writes_pgn(tmp_path, capsys):
    pgn = tmp_path / "games.pgn"
    main(["match", "greedy", "random", "--games", "2", "--max-moves", "40", "--pgn", str(pgn)])
    assert pgn.read_text().count("[Event") == 2
    assert "greedy vs random" in capsys.readouterr().out


@pytest.mark.skipif(not os.path.exists(LEARNBOT), reason="learnbot not built")
def test_uci_agent_with_learnbot():
    bot = make_agent(f"uci:{LEARNBOT} uci@50")
    try:
        g = play_game(bot, make_agent("random", 3), max_moves=20)
        assert len(list(g.mainline_moves())) > 0
    finally:
        bot["close"]()
