# Sentiment Analysis Project

## BiLSTM

The BiLSTM implementation is split by responsibility:

```text
BiLSTM/
├── train.py
├── evaluation.py
├── trained_models/
└── util/
    ├── data.py
    ├── model.py
    └── vokab.py
```

Train the model from the repository root with:

```bash
python3 -m BiLSTM.train
```

Evaluate it with:

```bash
python3 -m BiLSTM.evaluation
```

Checkpoints, the final model, vocabulary, and model metadata are written to
`BiLSTM/trained_models/`.
