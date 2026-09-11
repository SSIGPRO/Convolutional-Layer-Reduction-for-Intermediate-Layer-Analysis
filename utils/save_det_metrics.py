import pandas as pd
import torch
from pathlib import Path
from statistics import geometric_mean as geomean

# column prefix used in the reports of each detection metric
metric_prefixes = {
        'AUROC': 'AUC',
        'FPR@95': 'FPR',
        'DetError': 'DetError',
        'AURC': 'AURC',
        'E-AURC': 'E-AURC',
        }

# metrics aggregated with the geometric mean, the others with the arithmetic one
geometric_metrics = {'AUROC'}


def saved_scores(path, dataset, model, reduction=None):
    '''
    Return the score names (`analysis` column) already saved in `path` for a
    given `dataset` and `model`, so that they are not recomputed. If
    `reduction` is passed, only the scores saved for that reduction are
    returned.
    '''
    path = Path(path)
    if not path.exists():
        return set()

    df = pd.read_pickle(path)
    mask = (df['dataset'] == dataset) & (df['model'] == model)
    if reduction != None:
        mask &= df['reduction'] == reduction

    return set(df.loc[mask, 'analysis'])


def save_metrics(report, path, cols, dataset, model, reduction, analysis):
    path = Path(path)

    row = {
            'dataset': dataset,
            'model': model,
            'reduction': reduction,
            'analysis': analysis,
            **{c: report[c] for c in cols},
            }

    if path.exists():
        df = pd.read_pickle(path)
        mask = (
                (df['dataset']   == dataset)  &
                (df['model']     == model)     &
                (df['reduction'] == reduction) &
                (df['analysis']  == analysis)
                )
        if mask.any():
            df.loc[mask, cols] = [row[c] for c in cols]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])

    df.to_pickle(path)


def save_reports(reports, paths, dataset, model, reduction, analysis):
    '''
    Save the report of each detection metric to its own dataframe.

    Args:
    - reports (dict{str: dict}): reports keyed by the metric name, as returned by `make_reports()`.
    - paths (dict{str: str|pathlib.Path}): dataframe of each metric, keyed by the metric name.
    '''
    for metric, report in reports.items():
        prefix = metric_prefixes.get(metric, metric)
        cols = [f'{prefix} {_s}' for _s in ['OoD', 'AA', 'general']]
        save_metrics(report, paths[metric], cols+[f'{c} std' for c in cols], dataset, model, reduction, analysis)


def aggregate(values, geometric):
    '''
    Aggregate the values of a group of negative loaders and their standard deviation across the group.
    '''
    if len(values) == 0:
        return float('nan'), float('nan')

    std = torch.tensor(values).std().item() if len(values) > 1 else 0.

    if geometric:
        # guard against the zeros, so that the geometric mean is well defined
        return geomean([v if v >= 1e-4 else v + 1e-3 for v in values]), std

    return torch.tensor(values).mean().item(), std


def make_reports(res_ood, res_aa, score_name, metrics):
    '''
    Build the report of each detection metric from the `detection_metrics` outputs over the OoD and over the AA negative loaders. Each aggregate comes with a '<name> std', the standard deviation across the negative loaders it aggregates.

    Args:
    - res_ood, res_aa (pandas.DataFrame): outputs of `peepholelib.eval.detection_metrics.detection_metrics`.
    - score_name (str): score to report, as saved in the 'score name' column.
    - metrics (list[str]): detection metrics to report.

    Returns:
    - dict{str: dict}: report of each metric, keyed by the metric name.
    '''
    def _values(res, metric):
        _res = res[(res['score name'] == score_name) & (res['metric'] == metric)]
        return dict(zip(_res['neg loader'], _res['value']))

    reports = {}
    for metric in metrics:
        prefix = metric_prefixes.get(metric, metric)
        geometric = metric in geometric_metrics

        ood = _values(res_ood, metric)
        aa = _values(res_aa, metric)

        report = {f'{prefix} {k}': v for k, v in {**ood, **aa}.items()}
        report[f'{prefix} OoD'], report[f'{prefix} OoD std'] = aggregate(list(ood.values()), geometric)
        report[f'{prefix} AA'], report[f'{prefix} AA std'] = aggregate(list(aa.values()), geometric)
        report[f'{prefix} general'], report[f'{prefix} general std'] = aggregate(list(ood.values())+list(aa.values()), geometric)

        reports[metric] = report

    return reports
