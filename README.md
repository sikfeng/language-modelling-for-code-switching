# Language Modelling For Code-Switching

The codes used can be found here.

`data_generation/` contains the code used to extract the sentences from the datasets and formatting of dictionary and phoneme mapping.

`lstm_models/` contains the code used to train the LSTM model.

`mbert_models/` contains the code used to train the multilingual BERT (mBERT) models.

`mcnemar_test_mbert/` contains the code used to perform McNemar's Test on the mBERT models.

`mono_bert_models/` contains the code used to train monolingual BERT models on monolingual alternatives.

`rank_lstm_alts/` contains the code used to rank the scores of each type of alternative (code-switched, English, Chinese) on the LSTM models.

`rank_mbert_alts/` contains the code used to rank the scores of each type of alternative (code-switched, English, Chinese) on the mBERT models.
