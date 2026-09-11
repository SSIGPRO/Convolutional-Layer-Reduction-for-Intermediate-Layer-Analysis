# python stuff
import re
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

import pandas as pd

# Our stuff
from configs.common import *
from utils.save_det_metrics import metric_prefixes

# maps the macro of a row to the 'analysis' saved in the dataframe
analysis_macros = {
        'gls{macs}': 'MACS',
        'gls{dmd}': 'DMD',
        'gls{doctor}': 'DOC',
        'gls{relu}': 'Rel-U',
        'gls{vim}': 'ViM',
        'gls{fs}': 'FS',
        }

# maps the reduction macro of a sub-row to the 'reduction' saved in the dataframe
reduction_macros = {
        'avgpDimRed': 'avgpooling',
        'toeplitzDimRed': 'toeplitz',
        'kernelDimRed': 'kernel',
        }

# rows without a reduction sub-row (the compared scores) are saved with '-'
default_reduction = '-'

# maps the dataset to the macro used in the caption
dataset_macros = {
        'CIFAR100': '\\cifar',
        'ImageNet': '\\imagenet',
        }

# metrics whose best value is the highest one, the others are minimized
higher_is_better = {'AUROC'}

# column order matching the table layout, left to right
col_order = []
for model in ['VGG', 'MobileNet', 'ResNet', 'ConvNeXt']:
    for split in ['OoD', 'AA']:
        col_order.append((model, split))

# cells to be filled in the template, holding the value and its standard deviation
blank_value = '$._{\\pm .}$'

# tokens replaced in the template by the metric being filled
metric_token = 'DETMETRIC'
arrow_token = 'DETARROW'

def format_std(std):
    '''
    Standard deviation printed without its leading zero, e.g. 0.03 as .03
    '''
    return f'{std:.2f}'.removeprefix('0')

def fill(metric, df):
    '''
    Write one filled table per dataset for a detection metric.

    Args:
    - metric (str): detection metric, used in the caption and in the file name.
    - df (pandas.DataFrame): report of the metric, as saved by `save_reports()`.
    '''
    prefix = metric_prefixes.get(metric, metric)
    best_is_max = metric in higher_is_better

    # values[dataset][(analysis, reduction)][(model, split)] = (value, std)
    values = {}
    for _, row in df.iterrows():
        for _, split in col_order:
            col = f'{prefix} {split}'
            if not col in row or pd.isna(row[col]): continue
            entry = values.setdefault(row['dataset'], {}).setdefault((row['analysis'], row['reduction']), {})
            std = row.get(f'{col} std')
            entry[(row['model'], split)] = (row[col], 0. if pd.isna(std) else std)

    # best_dr[dataset][(analysis, model, split)] = best value across the reductions
    # best_col[dataset][(model, split)] = best value of the whole column
    best_dr = {}
    best_col = {}
    for dataset, ds_values in values.items():
        for (analysis, reduction), entry in ds_values.items():
            for (model, split), (value, _) in entry.items():
                # compared at the printed precision, so that the tied entries are all highlighted
                value = float(f'{value:.2f}')

                _col = best_col.setdefault(dataset, {}).get((model, split))
                if _col is None or (value > _col if best_is_max else value < _col):
                    best_col[dataset][(model, split)] = value

                # the compared scores have no reduction to be the best of
                if reduction == default_reduction: continue

                key = (analysis, model, split)
                _dr = best_dr.setdefault(dataset, {}).get(key)
                if _dr is None or (value > _dr if best_is_max else value < _dr):
                    best_dr[dataset][key] = value

    template_lines = (results_path/'det_metric_empty.tex').read_text().splitlines()

    for dataset, ds_values in values.items():
        current_analysis = None
        new_lines = []
        for line in template_lines:
            if '\\caption{' in line and dataset in dataset_macros:
                for _macro in dataset_macros.values():
                    line = line.replace(_macro, dataset_macros[dataset])

            line = line.replace(metric_token, metric)
            line = line.replace(arrow_token, '\\textuparrow' if best_is_max else '\\textdownarrow')

            line = re.sub(r'\\label\{([^}]*)\}', lambda m: f'\\label{{{m.group(1)} {metric} {dataset}}}', line)

            if blank_value in line:
                am = re.search(r'\\(gls\{\w+\})', line)
                if am and am.group(1) in analysis_macros:
                    current_analysis = analysis_macros[am.group(1)]

                rm = re.search(r'\\(\w+DimRed)', line)
                reduction = reduction_macros[rm.group(1)] if rm else default_reduction

                entry = ds_values.get((current_analysis, reduction), {})

                new_vals = re.findall(r'\$([^$]*)\$', line)
                for i, (model, split) in enumerate(col_order):
                    if not (model, split) in entry: continue

                    value, std = entry[(model, split)]
                    printed = float(f'{value:.2f}')
                    cell = f'{value:.2f}'

                    if best_col.get(dataset, {}).get((model, split)) == printed:
                        cell = f'\\mathbf{{{cell}}}'
                    if best_dr.get(dataset, {}).get((current_analysis, model, split)) == printed:
                        cell = f'\\underline{{{cell}}}'

                    new_vals[i] = f'{cell}_{{\\pm {format_std(std)}}}'

                it = iter(new_vals)
                line = re.sub(r'\$([^$]*)\$', lambda m: f'${next(it)}$', line)

            new_lines.append(line)

        out = results_path/f'{metric}_{dataset}.tex'
        out.write_text('\n'.join(new_lines) + '\n')
        print(f'Table saved in {out}')

if __name__ == "__main__":
    for _metric, _path in det_metrics_df_paths.items():
        if not Path(_path).exists():
            print(f'No report for {_metric} in {_path}, skipping')
            continue

        fill(_metric, pd.read_pickle(_path))
