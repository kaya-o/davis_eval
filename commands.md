Summary plots
```
python src/scripts/sale_ramdas_style_vis.py --result-dir results/20260528_000200_100_runs --hide-title
python src/scripts/selected_datapoints_vis.py --result-dir results/20260528_000200_100_runs --panels --n-runs 8
```

One off vs suite
```
python src/pipeline.py \
  --config configs/davis_experiment.json \
  --suite-name suite_20260527_231925_window_width_sweep \
  --run-name window_width_0_25
```

Create suite
```
 python src/scripts/create_suite.py k_sweep
 ```

python src/pipeline.py \
  --config configs/davis_experiment100.json \
  --suite-name suite_20260528_101421_k_sweep \
  --run-name k_100