# Convolutional Layer Dimensionality Reductions

Source code for the paper "A Convolutional Layer Activation Dimensionality Reduction for Out-of-Distribution and Adversarial Attack Detection Methods"

## Install the peepholelib

Our implementations are integrated within the peepholelib. 

```sh
git clone git@github.com:SSIGPRO/peepholelib.git
git checkout devel
```

## General syntax and structure

Folders and files containing experiments and configurations:
- `configs`: general configurations (paths, parameters, and auxiliary functions) for models, dimensionality reductions, and analysis.
- `fine_tune_models`: experiment for finetuning models.
- `datasets`: experiment for parsing dataset and applying adversarial attacks.
- `corevectors`: experiment for computing the corevectors.
- `tuning`: experiment for tuning the OoD and AA detection methods.
- `det_metrics`: experiments for computing the detection metrics, both for the best configuration from tuning (`xp_peepholes.py`) and for the baseline detection methods (`xp_others.py`).
- `eval`: making plots, computing AUCs, and generating result tables.
- `temp_results`: plots and filled LaTeX tables are saved here.
- `utils`: auxiliary implementations.
- `Makefile`: shortcuts for running all tests.

The general syntax for the experiments follows:
```sh
python <folder>/<experiment name>.py -m <Model> -r <reduction> -a <analysis> -ds <dataset> -d <directory>
```
Where:
- `<Model> = [VGG|MobileNet|ResNet|ConvNeXt]` is the model to evaluate.
- `<reduction> = [kernel|toeplitz|avgpooling]` is the dimensionality reduction to use.
- `<dataset> = [CIFAR100|ImageNet]` is the dataset to use.
- `<analysis> = [MACS|DMD]` is the OoD and AA detection method.
- `<directory>` is a path in your system to save the corevectors, peepholes, and other generated data.

## Analysis methods

- **[MACS](https://arxiv.org/abs/2512.19472)** — Mixture-of-Gaussians-based classifier (protoclass score) operating on corevectors.
- **[DMD](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)** — Deep Mahalanobis Distance detector; scores samples by their Mahalanobis distance to per-class Gaussians fitted on corevectors, with optional input pre-processing controlled by a `magnitude` hyperparameter.

## Using the Makefile

To run all combinations of models, reduction and analysis, one can use:
```sh
make xp_datasets
make xp_corevectors
make xp_tuning
make xp_det_metrics_ph
make xp_det_metrics_others
```
The `xp_det_metrics_ph` rule runs `det_metrics/xp_peepholes.py` for every reduction and analysis, and `xp_det_metrics_others` runs `det_metrics/xp_others.py` for the baseline detection methods.

## Finetuning models

To finetune the models to `CIFAR100` use:
```sh
python fine_tune_models/xp_finetune_model.py -m <model>
```bran

## Reproducing results

To plot the `AUC`s scatter plot from the paper run:
```sh
python eval/aucs.py -d <directory>
```

To check model accuracy on a dataset run:
```sh
python eval/model_acc.py -m <Model> -ds <dataset> -d <directory>
```

To fill the best-configuration LaTeX table (requires tuning results) run:
```sh
python eval/fill_table.py -d <directory>
```
This reads `temp_results/blank_table.tex` as a template and writes one filled `temp_results/bestConfigs_<dataset>.tex` per dataset.

To fill the detection metrics LaTeX tables (requires the reports produced by `make xp_det_metrics_ph` and `make xp_det_metrics_others`) run:
```sh
python eval/fill_det_metrics.py -d <directory>
```
This reads `temp_results/det_metric_empty.tex` as a template and writes one filled `temp_results/<metric>_<dataset>.tex` per metric and dataset, for every metric in `det_metrics`. Each cell holds the average metric with its standard deviation as a subscript. The best value across the reductions is underlined for `MACS` and `DMD` independently, and the best value of each column is highlighted in bold. Entries missing from the results are left as `$._{\pm .}$`.

## Problems Running?
The `peepholelib` has a considerable number of dependencies and the scripts might break depending on their versions. Check the `requirements.txt` file with the main packages versions used in our experiments.

Note we use the `devel` branch of `peepholelib`, whose interface is not stable (frequent updates). The exact commit we use is `c22a041`.
