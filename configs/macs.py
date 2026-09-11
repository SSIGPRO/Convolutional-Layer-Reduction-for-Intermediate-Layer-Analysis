# Torch stuff
from torch import linspace, int32

# Ray Stuff
from ray.tune import choice 

# Peepholelib stuff
from peepholelib.peepholes.classifiers.tgmm import GMM as Driller
from peepholelib.scores.protoclass import ProtoClassScore

bs_analysis_scale = 2**5

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    configs = kwargs['configs']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        n_clusters = configs[_l]['n_clusters']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{n_clusters}',
                'target_module': _l,
                'nl_classifier': n_clusters,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'cls_kwargs': {
                    'covariance_regularization': 1e-4,
                    'convergence_tolerance': 1e-2
                    },
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        if args.dataset == 'CIFAR100':
            _l['n_clusters'] = choice(linspace(50, 500, 10, dtype=int32).numpy().tolist())
        if args.dataset == 'ImageNet':
            _l['n_clusters'] = choice(linspace(50, 5000, 10, dtype=int32).numpy().tolist())
        
    configs['model'] = args.model
    configs['reduction'] = args.reduction
    configs['analysis'] = args.analysis
    configs['dataset'] = args.dataset
    return configs

def get_scores(**kwargs):
    '''
    Return the analysis scores along with the arguments of their `fit()` and `compute()`, so that the scripts can fit and compute them uniformly.

    Args:
    - path (str|pathlib.Path): folder where the scores are saved.
    - name (str): name of the score.
    - ds_name (str), model (str): dataset and model names, used to build the loader keys.
    - loaders (list[str]): loaders to score.
    - proto_threshold (float): model's confidence threshold to select the protoclass samples. Defaults to 0.9.

    Returns:
    - list[dict]: one entry per score, with the score in 'score' and its specific arguments in 'fit' and 'compute'.
    '''
    path = kwargs['path']
    name = kwargs['name']
    ds_name = kwargs['ds_name']
    model = kwargs['model']
    loaders = kwargs['loaders']
    proto_threshold = kwargs.get('proto_threshold', 0.9)

    return [{
        'score': ProtoClassScore(path=path, name=name),
        'fit': {
            'fit_key': f'{ds_name}-train-{model}',
            'proto_threshold': proto_threshold,
            },
        'compute': {
            'loaders': loaders,
            },
        }]
