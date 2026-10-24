# chess auto agent

lets chess bots play each other on your laptop, saves the games, and tells you who is better (in elo). no internet, nothing online.

## agents

| name | what it does |
| --- | --- |
| `random` | any legal move |
| `greedy` | takes the biggest piece it can, else random |
| `minimax:N` | alpha beta search N moves deep, my own eval (material + piece square tables), captures are searched till its calm |
| `uci:<command>` | any UCI engine: stockfish, or my **learnbot** from the chess-learning-bot repo. `@200` = 200 ms per move, `=name` = its name |

## tournament

everyone vs everyone, 4 games per pair, both colours:

| # | agent | points | elo |
| --- | --- | --- | --- |
| 1 | minimax:2 | 19 / 20 | +756 |
| 2 | minimax:1 | 16.5 / 20 | +619 |
| 3 | learnbot-trained | 11.5 / 20 | +402 |
| 4 | learnbot-zero | 7 / 20 | +194 |
| 5 | greedy | 4 / 20 | +90 |
| 6 | random | 2 / 20 | 0 |

(elo compared to random. only 4 games per pair, so the numbers are rough)

what i learned from it:
- the self taught learnbot really got better (zero -> trained = +208 here), but it is still weaker than a 1 move minimax with hand made piece tables. 16 learned numbers are not enough yet.
- greedy vs random is almost always a draw: both trade everything and nobody can mate.

## use

```
pip install -r requirements.txt

python -m src.agent match minimax:2 random --games 10
python -m src.agent match minimax:2 greedy --games 20 --pgn games/out.pgn
python -m src.agent tournament random greedy minimax:1 minimax:2 --games 4
python -m src.agent play minimax:2                 # you vs the bot
```

with learnbot (build it first in the chess-learning-bot repo):

```
python -m src.agent match "uci:../chess-learning-bot/build/learnbot uci --weights ../chess-learning-bot/models/best.txt@100=learnbot" minimax:1
```

works the same with stockfish: `uci:stockfish@100`.

games start from 10 short openings (each played with both colours) so they are not all the same. after 200 half moves a game is called a draw.

## elo

`elo difference = -400 * log10(1/score - 1)`. with a +/- range so you know how sure it is. if one side wins every game we can only say "at least +X", so it says that. the tournament fits ratings for everyone together (simple bradley-terry style fit).

## tests

```
pytest
```

the learnbot test runs only if `../chess-learning-bot/build/learnbot` exists.

## todo

- time controls instead of fixed ms per move
- opening book from a pgn file
- run games in parallel
