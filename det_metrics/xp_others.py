# python stuff
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/ConvRed').as_posix())

from functools import partial
from filelock import FileLock

# torch stuff
import torch
from cuda_selector import auto_cuda

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.eval.detection_metrics import detection_metrics

# --- softmax scores
from peepholelib.scores.doctor import DOCTORScore
from peepholelib.scores.relu import RelUScore

# --- pre logits based scores
from peepholelib.scores.vim import VIMScore

# --- input based scores
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector as FSD
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, bit_depth_torch, MedianPool2d
from peepholelib.scores.feature_squeezing import FeatureSqueezingScore

from configs.common import *
from utils.get_best_configs import test_configs
from utils.save_det_metrics import saved_scores, make_reports, save_reports

if __name__ == "__main__":
    print(f'{args}')
    lock_file = '../locks/peepholes.cuda.lock'
    lock = FileLock(lock_file)
    with lock.acquire(timeout=-1):
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

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

    # the train loader is needed to fit Rel-U and ViM
    loaders = [k for k in get_loaders(ood_datasets) if 'test' in k or k in [f'{args.dataset}-val', f'{args.dataset}-train']]
    inference_names = get_inference_names(ood_datasets)
    transforms = get_transforms(ood_datasets)

    #--------------------------------
    # Dataset
    #--------------------------------
    datasets = ParsedDataset(
            path = ds_path,
            )

    with datasets as ds:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        score_loaders = [k for k in ds._dss.keys() if 'test' in k]

        _saved = saved_scores(saved_df_path, args.dataset, args.model)
        print('Scores already saved: ', sorted(_saved))

        scores = []

        ## SoftMax based scores
        if not 'DOC' in _saved:
            doc = DOCTORScore(path=others_scores_path, name='DOC')
            doc.compute(
                    datasets = ds,
                    model = model,
                    loaders = score_loaders,
                    batch_size = 2**10,
                    verbose = verbose,
                    )
            scores.append(doc)

        if not 'Rel-U' in _saved:
            relu = RelUScore(path=others_scores_path, name='Rel-U')
            if not relu.load():
                relu.fit(
                        datasets = ds,
                        fit_key = f'{args.dataset}-train-{args.model}',
                        verbose = verbose,
                        )
            relu.compute(
                    datasets = ds,
                    loaders = score_loaders,
                    verbose = verbose,
                    )
            scores.append(relu)

        ## Pre logits based score
        if not 'ViM' in _saved:
            vim = VIMScore(path=others_scores_path, name='ViM')
            if not vim.load():
                vim.fit(
                        model = model,
                        datasets = ds,
                        output_layer = output_layer,
                        fit_key = f'{args.dataset}-train-{args.model}',
                        batch_size = 2**10,
                        n_threads = n_threads,
                        verbose = verbose,
                        )
            vim.compute(
                    model = model,
                    datasets = ds,
                    output_layer = output_layer,
                    loaders = score_loaders,
                    batch_size = 2**10,
                    n_threads = n_threads,
                    verbose = verbose,
                    )
            scores.append(vim)

        ## Input based score
        if not 'FS' in _saved:
            fsd = FSD(
                model = model,
                prepro_dict = {
                    'median': MedianPool2d(kernel_size=3, stride=1, padding=1),
                    'bit_depth': partial(bit_depth_torch, bits=5),
                    'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32, device=device),
                    }
                )

            fs = FeatureSqueezingScore(path=others_scores_path, name='FS')
            fs.compute(
                    datasets = ds,
                    loaders = score_loaders,
                    detector = fsd,
                    batch_size = 2**10,
                    verbose = verbose,
                    )
            scores.append(fs)

        #--------------------------------
        # AUCs
        #--------------------------------
        score_names = [_s.name for _s in scores]
        print('Computing AUCs for: ', score_names)

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

        reports = {}
        for score_name in score_names:
            reports[score_name] = make_reports(res_ood, res_aa, score_name, det_metrics)

            for _metric, _report in reports[score_name].items():
                print(f'{_metric} report {score_name}: ', _report)

    with lock.acquire(timeout=-1):
        for score_name in score_names:
            save_reports(
                    reports[score_name],
                    det_metrics_df_paths,
                    dataset   = args.dataset,
                    model     = args.model,
                    reduction = '-',
                    analysis  = score_name,
                    )
