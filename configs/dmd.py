# Ray Stuff
from torch import linspace
from ray.tune import choice 

# Peepholelib stuff
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as Driller
from peepholelib.scores.dmd import DMDScore

bs_analysis_scale = 2**-2

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    model = kwargs['model']
    configs = kwargs['configs']
    act_parser = kwargs['act_parser']
    save_input = kwargs['save_input']
    save_output = kwargs['save_output']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        mag = configs[_l]['magnitude']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{mag}', 
                'target_module': _l,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'model': model,
                # divide by 1000 to avoid large file names
                'magnitude': mag/1000,
                'std_transform': [0.229, 0.224, 0.225],
                'act_parser': act_parser,
                'save_input': save_input,
                'save_output': save_output,
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        # mag is divided by 1000 at dmd to avoid large file names
        _l['magnitude'] = choice(linspace(1, 10, 10).numpy().tolist())
    configs['model'] = args.model
    configs['reduction'] = args.reduction
    configs['analysis'] = args.analysis
    configs['dataset'] = args.dataset
    return configs

def get_scores(**kwargs):
    '''
    Return the analysis scores along with the arguments of their `fit()` and `compute()`, so that the scripts can fit and compute them uniformly. One regressor is trained for each negative loader, and the loader it was trained against is kept in the 'calib key' column of the score.

    Args:
    - path (str|pathlib.Path): folder where the scores are saved.
    - name (str): name of the score.
    - ds_name (str), model (str): dataset and model names, used to build the loader keys.
    - neg_loaders (dict{str: list[str]}): negative test loaders mapped to the negative train loaders used to train their regressor.

    Returns:
    - list[dict]: one entry per score, with the score in 'score' and its specific arguments in 'fit' and 'compute'.
    '''
    path = kwargs['path']
    name = kwargs['name']
    ds_name = kwargs['ds_name']
    model = kwargs['model']
    neg_loaders = kwargs['neg_loaders']

    return [{
        'score': DMDScore(path=path, name=name),
        'fit': {
            'pos_train_loader': f'{ds_name}-val-{model}',
            'neg_loaders': neg_loaders,
            },
        'compute': {
            'pos_test_loader': f'{ds_name}-test-{model}',
            },
        }]
