# Packaging changes

The original project has not been modified. Model, loss, metric and data-processing implementations are retained. No historical model weights or prediction results are included.

Added release documentation, dependency versions, split indices, a command-line wrapper, a smoke check and file manifests. The wrapper supplies explicit run-time paths/settings to existing entry points. Existing source scripts keep their original USER_CFG defaults: use run.py for the documented workflow.

Only historical machine-path examples in comments/docstrings were replaced by portable examples in these copied source files:
- src/figs_paper/fig1_icons.py
- src/figs_paper/fig6_typical.py
- src/figs_paper/run_all.py
- src/figs_paper/run_all_final.py
- src/infer/run_ablation_eval.py
- src/infer/run_ablation_eval_s04.py
- src/train/run_ablation_train_s04.py

The legacy duplicate dataset cache, figures, outputs directories, code backups, editor settings and bytecode are excluded. Dataset files are copied without changing their numerical content. The original raw data layout calls the combined 1500-sample training/validation pool `train`; split_indices.csv disambiguates this.

No data-generation executable or MODFLOW simulation project was present in the selected project source tree. The package starts from the included simulated fields/heads and observation files; it does not claim to regenerate the original groundwater simulations from scratch. Historic run-specific configurations cannot be recovered completely from the current defaults alone. Original checkpoints, predictions and experiment logs are intentionally excluded.
