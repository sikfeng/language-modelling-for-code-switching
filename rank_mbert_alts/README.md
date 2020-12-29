# Ranking different types of alternate sentences with the BERT model

The file containing the alternative sentences should be placed in `data/`, and the models should be placed in `models/`.

`evaluate_model.py` will write to the files `val_scores.txt` and `test_scores.txt`, which are used by `rank_alts.py` to get the rankings of the types of alternative sentences.
