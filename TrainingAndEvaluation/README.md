# Training and Evaluation

XGBoost classifier for three-class sentiment (Negative / Neutral / Positive) from MediaPipe body and face motion features extracted from DGS fairy tale videos.

---

## Input

All notebooks expect the aggregated feature file produced by the data pipeline:

```
Aggregated/
  AllTales-Aggregated.csv       # merged features + LLM sentiment labels
  AllFront_features.csv         # per-tale feature CSVs (merged by aggregate.py)
```

---

## Notebooks

### `DGS_stratkfold_sentiment_f.ipynb`
Repeated Stratified Group K-Fold cross-validation using XGBoost. Groups are defined by tale/story, so no tale appears in both train and test splits.

Outputs to `Aggregated/results/<timestamp>-RSGKF-sentiment/`:
| File | Contents |
|---|---|
| `*-RSGKF_folds.csv` | Per-fold metrics (accuracy, balanced accuracy, macro/weighted F1) |
| `*-RSGKF_predictions.csv` | Per-segment predictions across all folds |
| `*-RSGKF_summary.csv` | Mean ± std across folds |
| `*-FeatImp_Fold_*.csv` | Per-fold feature importances |
| `*-FeatImp_mean.csv` | Aggregated mean feature importance |
| Confusion matrix plots | Per-fold and aggregate |

### `DGS_GridSearchCV.ipynb` — Hyperparameter search
Grid search over XGBoost hyperparameters using Stratified Group K-Fold (4-split inner CV). Saves the best parameters and then evaluates the best estimator.

Outputs to `Aggregated/results/<timestamp>-GridSearch-sentiment/`:
| File | Contents |
|---|---|
| `*-GridSearch_cv_results.csv` | Full grid results ranked by macro F1 |
| `*-GridSearch_best_params.json` | Best hyperparameter set |
| `*-FinalModel_metrics.csv` | Metrics of best estimator on full search set |
| `*-FeatImp_mean.csv` | Feature importances of best model |

### `DGS_cross_validation.ipynb` — Fixed-feature evaluation
5-fold cross-validation using a manually curated subset of top features (selected from prior importance analysis).

Outputs to `Aggregated/results/<timestamp>-5FoldCV/`:
| File | Contents |
|---|---|
| `*-FeatureImportances_All.csv` | Mean importance across folds |
| Top-20 feature importance plot | Bar chart with std error bars |

---

## Workflow

1. Run `DGS_stratkfold_sentiment_f.ipynb` to get baseline results and feature importances.
2. Use importances to inform the parameter grid, then run `DGS_GridSearchCV.ipynb` to tune.
3. Run `DGS_cross_validation.ipynb` to evaluate a curated feature subset.

---

## Reference

> Nunnari, F., Jain, S., & Gebhard, P. (2026). *Sentiment Analysis of German Sign Language Fairy Tales*. arXiv:2604.16138.
