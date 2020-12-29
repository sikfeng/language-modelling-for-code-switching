# Data generation

`datasets/` contains the datasets where the sentences are extracted from.

`dictionaries/` contains the original dictionary files ([CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), [CC-CEDICT](https://cc-cedict.org/wiki/)).

`scripts/` contains the scripts to extract and format the sentences and dictionaries.

`extracted/` is where the formatted sentences and dictionaries are stored.

The alternative sentences can be generated using [this method](https://github.com/gonenhila/codeswitching-lm/tree/5a8b843a7ab952d62de76d50f77bb75f38897620/evaluation_dataset) using the files in `extracted/`. The plain text files containing the sentences can be split into smaller files if wanted.
