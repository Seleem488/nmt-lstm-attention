# Assignment 3 - Neural Machine Translation with LSTM and Attention

Natural Language Processing - 

English → German neural machine translation with two PyTorch Seq2Seq models:

1. **Baseline LSTM** - the encoder's final hidden/cell state initializes the decoder (no attention).
2. **BiLSTM + Bahdanau attention** - a bidirectional encoder exposes every source state, and the decoder computes masked additive attention at each step.

Both models are trained on the same data, with the same optimizer and epoch budget, and evaluated on the same held-out test set.

## Files

| File | Description |
|---|---|
| `_NMT_LSTM_Attention.ipynb` | Main notebook: preprocessing, both models, training, loss curves, BLEU/ROUGE-L, length analysis, attention heatmaps, error analysis |
| `_Report.pdf` | Written report (pipeline, architectures, results, heatmap and error analysis, limitations) |
| `GERMAN_ENGLISH_TRANSLATION.csv` | Dataset (columns `ENGLISH`, `GERMAN`; 152,820 pairs) - place it next to the notebook |
| `README.md` | This file |

Running the notebook also creates:

| Output | Content |
|---|---|
| `checkpoints/` | Best-validation checkpoints (`baseline_best.pt`, `attention_best.pt`) |
| `artifacts/` | Final model weights, vocabularies (`src_vocab.pkl`, `tgt_vocab.pkl`), `run_config.json` |
| `evaluation_results.csv` | BLEU-1..4 and ROUGE-L for both models |
| `length_analysis.csv` | BLEU-4 for short (3-6 words) vs. long (12-18 words) sentences |
| `qualitative_examples.csv` | The 8-example side-by-side table |

## Requirements

- Python 3.9+
- `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `nltk`, `rouge-score`

```bash
pip install torch numpy pandas matplotlib seaborn tqdm nltk rouge-score
```

A CUDA GPU is strongly recommended. The saved run used an NVIDIA RTX 3050 Laptop GPU.

## How to run

1. Put `GERMAN_ENGLISH_TRANSLATION.csv` in the same folder as the notebook.
2. In the data-loading cell, set the path to the local file:
   ```python
   DATA_PATH = "GERMAN_ENGLISH_TRANSLATION.csv"
   ```
3. Run all cells from top to bottom (`Kernel → Restart & Run All`).

Approximate runtime on the RTX 3050 laptop GPU: about 2 minutes per epoch for the baseline and about 4 minutes per epoch for the attention model (15 epochs each), plus about 4.5 minutes of greedy decoding on the test set. On CPU it will be much slower; lower `EPOCHS` or `MAX_LEN` in the configuration cell if needed.

The random seed is fixed (`SEED = 42`), but GPU results can still differ slightly between machines.

## Pipeline summary

- **Cleaning:** lowercase, keep letters (including umlauts and ß) and apostrophes, drop punctuation and digits, collapse whitespace; remove 224 duplicate pairs → 152,596 pairs.
- **Split:** seeded random 80/10/10 → 122,076 train / 15,259 validation / 15,261 test.
- **Vocabulary:** word-level, built from the training set only, `min_freq = 2` → 9,332 source / 15,007 target types, with `<pad>`, `<sos>`, `<eos>`, `<unk>`.
- **Padding:** dynamic per batch; encoders use `pack_padded_sequence`; the loss ignores `<pad>`; maximum length 30.
- **Training:** embedding/hidden 256, dropout 0.25, Adam (lr 1e-3), gradient clipping 1.0, teacher forcing 0.5, 15 epochs, best checkpoint chosen by validation loss.
- **Decoding:** greedy.

## Results (test set, 15,261 pairs)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Baseline LSTM (11.1M params) | 0.5335 | 0.3966 | 0.2979 | 0.2251 | 0.5512 |
| BiLSTM + Bahdanau (24.2M params) | 0.5894 | 0.4584 | 0.3600 | 0.2842 | 0.6021 |

BLEU-4 by source length:

| Source length | N | Baseline | Attention |
|---|---|---|---|
| Short (3-6 words) | 8,282 | 0.2989 | 0.3394 |
| Long (12-18 words) | 698 | 0.0881 | 0.1525 |

Best validation loss: baseline 2.872 (epoch 11-12), attention 2.635 (epoch 5). The attention model overfits after epoch 5, and the best checkpoint is restored before evaluation.

## Known limitations

- Random pair-level split: some English sentences appear with several German translations, so the same source can occur in both train and test (may slightly inflate scores).
- Single reference per sentence, greedy decoding, one training run.
- Word-level vocabulary produces `<unk>` for rare words; no coverage mechanism, so attention can repeat or skip words.

See `Report.pdf` for the full analysis.
