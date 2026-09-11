# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.eval.detection_metrics import detection_metrics

from configs.common import *
from utils.get_best_configs import test_configs
from utils.save_det_metrics import saved_scores, make_reports, save_reports

if __name__ == "__main__":
    print(f'{args}')

    _saved = saved_scores(saved_df_path, args.dataset, args.model, reduction=args.reduction)
    if args.analysis in _saved:
        print(f'AUCs for {args.dataset} {args.model} {args.reduction} {args.analysis} already saved in {saved_df_path}. Skipping.')
        quit()

    lock_file = '../locks/peepholes.cuda.lock'
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        bs = int(bs_base*bs_model_scale*bs_red_scale*bs_analysis_scale)

        #------------------
        # Model 
        #------------------
        model = ModelWrap(
                model = Model(weights = pre_train_weights.DEFAULT),
                target_modules = target_layers,
                device = device
                )
    
    # in the non-imagenet case, we overwrite the model's
    # last layer and its weights
    if update_output:
        model.update_output(
                output_layer = output_layer, 
                to_n_classes = n_classes,
                overwrite = True 
                )
                                                
        model.load_checkpoint(
                path = model_path,
                name = model_name,
                verbose = True 
                )
                                                         
    model.set_normalizer(
            mean = normalization_mean,
            std = normalization_std
            )

    #--------------------------------
    # Dataset 
    #--------------------------------
    datasets = ParsedDataset(
            path = ds_path,
            )
    
    #--------------------------------
    # Corevectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            model = model,
            )
    
    #--------------------------------
    # Peepholes
    #--------------------------------
    hyperps = test_configs(model._target_modules, hyper_params_file)

    # create phs names from configs
    # same as tuning
    phs_names = {} 
    for _l, _c in hyperps.items():
        if type(_c) == dict:
            phs_names[_l] = ''
            for _cn, _cv in _c.items():
                phs_names[_l] += f'{_cn}{_cv}'

    peepholes = Peepholes(
            path = phs_path,
            device = device
            )

    loaders = get_loaders(ood_datasets)
    inference_names = get_inference_names(ood_datasets)
    transforms = get_transforms(ood_datasets)

    with datasets as ds, corevecs as cv, peepholes as ph:
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

        ph.load_only(
                loaders = list(ds._dss.keys()),
                names = phs_names,
                verbose = verbose 
                )

        scores = []
        for _score in get_scores(
                path = scores_path,
                name = args.analysis,
                ds_name = args.dataset,
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
                metrics = det_metrics,
                verbose = verbose
                )

        res_aa = detection_metrics(
                scores = scores,
                datasets = ds,
                pos_loader = get_pos_loader(),
                neg_loaders = get_neg_loaders_aa(atk_names),
                metrics = det_metrics,
                filter_key = 'attack_success',
                verbose = verbose
                )

        reports = make_reports(res_ood, res_aa, args.analysis, det_metrics)

        for _metric, _report in reports.items():
            print(f'{_metric} report: ', _report)

    with lock.acquire(timeout=-1):
        save_reports(
                reports,
                det_metrics_df_paths,
                dataset   = args.dataset,
                model     = args.model,
                reduction = args.reduction,
                analysis  = args.analysis,
                )
