# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

import pandas as pd

# torch stuff
import torch
from cuda_selector import auto_cuda

# Our stuff
import peepholelib
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.eval.detection_metrics import detection_metrics
from configs.common import *
from utils.save_det_metrics import make_reports

# Tuner
from functools import partial
import ray
from ray import tune
from ray import train
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import get_device as ray_get_device
import ray.cloudpickle as pickle

ray.init(runtime_env = {"py_modules": [peepholelib]})

def peephole_wrap(config, **kwargs):
    print(f'Running config: {config}') 

    model_path = kwargs['model_path']
    model_name = kwargs['model_name']
    ds = kwargs['datasets']
    cv = kwargs['corevectors']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    bs = kwargs['batch_size']
    ds_name = kwargs['dataset_name']
    verbose = kwargs['verbose']

    _device = ray_get_device() 

    # concatenate config for creating unique ph name
    phs_names = {} 
    for _l, _c in config.items():
        if type(_c) == dict:
            phs_names[_l] = ''
            for _cn, _cv in _c.items():
                phs_names[_l] += f'{_cn}{_cv}'

    #--------------------------------
    # instances
    #--------------------------------
    model = ModelWrap(
            model = Model(weights = pre_train_weights.DEFAULT),
            target_modules = target_layers,
            device = _device
            )

    if update_output:
        model.update_output(
                output_layer = output_layer,
                to_n_classes = n_classes,
                overwrite = True
                )

        model.load_checkpoint(
                name = model_name,
                path = model_path,
                verbose = verbose
                )
    
    model.set_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    drillers_kwargs = get_drillers_kwargs(
            path = drill_path,
            name = drill_name,
            target_layers = target_layers,
            nl_model = n_classes,
            model = model,
            configs = config,
            act_parser = act_parser, 
            save_input = save_input,
            save_output = save_output,
            device = _device
            ) 

    # instantiate the drillers
    drillers = {}
    for _l in target_layers:
        drillers[_l] = Driller(
                **drillers_kwargs[_l],
                reducer = Reducer(
                    path = svds_path,
                    model = model,
                    layer = _l,
                    cv_dim = drillers_kwargs[_l]['n_features'],
                    verbose = verbose
                    )
                )

        if not drillers[_l].load():
            drillers[_l].fit(
                    datasets = ds,
                    corevectors = cv,
                    loader = f'{ds_name}-train-'+args.model,
                    verbose=verbose
                    )
            drillers[_l].save()

    peepholes = Peepholes(
            path = ph_path,
            device = _device
            )

    with peepholes as ph:
        ph.get_peepholes(
            datasets = ds,
            corevectors = cv,
            target_modules = target_layers,
            batch_size = bs,
            drillers = drillers,
            names = phs_names,
            retry_load_time = 60,
            n_threads = 1,
            verbose = verbose 
            )

        # Evaluation
        # the scores are saved along the trial's peepholes, so that the trials do not share them
        scores = []
        for _score in get_scores(
                path = ph_path/'scores',
                name = args.analysis,
                ds_name = ds_name,
                model = args.model,
                loaders = list(ds._dss.keys()),
                neg_loaders = get_neg_loaders_fit(ood_datasets, atk_names),
                proto_threshold = proto_threshold,
                ):
            score = _score['score']

            if not score.load():
                score.fit(
                        datasets = ds,
                        peepholes = ph,
                        target_modules = target_layers,
                        verbose = verbose,
                        **_score['fit']
                        )

            score.compute(
                    datasets = ds,
                    peepholes = ph,
                    target_modules = target_layers,
                    verbose = verbose,
                    **_score['compute']
                    )

            scores.append(score)

        res_ood = detection_metrics(
                scores = scores,
                pos_loader = get_pos_loader(),
                neg_loaders = get_neg_loaders_ood(ood_datasets),
                metrics = ['AUROC'],
                verbose = verbose
                )

        res_aa = detection_metrics(
                scores = scores,
                datasets = ds,
                pos_loader = get_pos_loader(),
                neg_loaders = get_neg_loaders_aa(atk_names),
                metrics = ['AUROC'],
                filter_key = 'attack_success',
                verbose = verbose
                )

        # the configurations are still selected by their AUC, see get_best_config()
        report = make_reports(res_ood, res_aa, args.analysis, ['AUROC'])['AUROC']

        print('Report: ', report)

        train.report(report)
    return 

if __name__ == "__main__":
    print(f'{args}') 
    use_cuda = torch.cuda.is_available()

    #--------------------------------
    # Instances 
    #--------------------------------

    # dummy model to get instances of target layers
    dummy_model = ModelWrap(
            model = Model(),
            target_modules = target_layers,
            )
    
    dummy_model.set_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    datasets = ParsedDataset(
            path = ds_path,
            )

    corevecs = CoreVectors(
            path = cvs_path,
            )

    loaders = get_loaders(ood_datasets)
    inference_names = get_inference_names(ood_datasets)
    transforms = get_transforms(ood_datasets)

    with datasets as ds, corevecs as cv:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        cv.load_only(
                loaders = list(ds._dss.keys()),
                names = cvs_names,
                verbose = verbose 
                ) 

        #--------------------------------
        # Tunning 
        #--------------------------------
        red_kwargs = get_reducer_kwargs(
                dummy_model._target_modules
                ) 
        param_space = reduction_param_space(
                red_kwargs
                )
        param_space = analysis_param_space(
                configs = param_space,
                args = args
                )

        # resources per trial, leave gpu=1
        if use_cuda:
            resources = {"cpu": 8, "gpu": 1}
        else:
            resources = {"cpu": 8}

        if hyper_params_file.exists():
            print('Already tunned parameters found in %s. Runing agains and appending results.'%(hyper_params_file.as_posix()))
            _prev_results_df = pd.read_pickle(hyper_params_file)
        else:
            _prev_results_df = None

        searcher = OptunaSearch(metric=['AUC general'], mode=['max'])
        algo = ConcurrencyLimiter(searcher, max_concurrent=6)
        scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric='AUC general', mode='max') 

        trainable = tune.with_resources(
                partial(
                    peephole_wrap,
                    model_path = model_path,
                    model_name = model_name,
                    datasets = ds,
                    corevectors = cv,
                    ph_path = phs_path,
                    ph_name = phs_name,
                    batch_size = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale),
                    dataset_name = args.dataset,                
                    verbose = verbose
                    ),
                resources 
                )

        _existing_runs = sorted(tune_storage_path.glob('peephole_wrap_*'))
        _restore_path = _existing_runs[0].as_posix() if _existing_runs else None

        if _restore_path is not None and tune.Tuner.can_restore(_restore_path):
            print(f'Restoring existing tuning run from {_restore_path}')
            tuner = tune.Tuner.restore(
                    _restore_path,
                    trainable = trainable,
                    restart_errored = True,
                    param_space = param_space,
                    )
        else:
            tuner = tune.Tuner(
                    trainable,
                    tune_config = tune.TuneConfig(
                        search_alg = algo,
                        num_samples = tune_num_samples,
                        scheduler = scheduler,
                        ),
                    run_config = train.RunConfig(
                        storage_path = tune_storage_path
                        ),
                    param_space = param_space,
                    )

        result = tuner.fit()

        results_df = result.get_dataframe()
        # remove config/ from the DF coluns names
        _cn_map = {_cn: _cn.replace('config/', '') for _cn in results_df.columns if 'config/' in _cn}
        results_df = results_df.rename(columns=_cn_map)
        if _prev_results_df is not None:
            results_df = pd.concat([_prev_results_df, results_df], ignore_index=True)
        print('results: ', results_df)

        results_df.to_pickle(hyper_params_file)
    
