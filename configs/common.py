# Python stuff
from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['VGG','MobileNet', 'ResNet', 'ConvNeXt'], default='VGG')
parser.add_argument('-r', '--reduction', choices=['avgpooling', 'toeplitz', 'kernel'], default='kernel')
parser.add_argument('-a', '--analysis', choices=['MACS', 'DMD', 'MRC'], default='MACS')
parser.add_argument('-ds', '--dataset', choices=['CIFAR100', 'ImageNet'], default='CIFAR100')
parser.add_argument('-d', '--data_path', default=Path.cwd()/'../data')
args = parser.parse_args()

# import configs
if args.model == 'VGG':
    from configs.vgg import *
elif args.model == 'MobileNet':
    from configs.mobilenet import *
elif args.model == 'ResNet':
    from configs.resnet import *
elif args.model == 'ConvNeXt':
    from configs.convnext import *

if args.reduction == 'avgpooling':
    from configs.avgpooling import *
elif args.reduction == 'toeplitz':
    from configs.toeplitz import *
elif args.reduction == 'kernel':
    from configs.kernel import *

if args.analysis == 'MACS':
    from configs.macs import *
elif args.analysis == 'DMD':
    from configs.dmd import *
elif args.analysis == 'MRC':
    from configs.mrc import *

if args.dataset == 'CIFAR100':
    from configs.cifar100 import *
elif args.dataset == 'ImageNet':
    from configs.imagenet import *

from peepholelib.datasets.functional.transforms import TransformWrap
from peepholelib.datasets.functional.transforms import means as _means, stds as _stds
normalization_mean = _means[args.dataset]
normalization_std = _stds[args.dataset]

from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.AutoAttack import myAutoAttack

#--------------------------------
# Paths and Definitions
#--------------------------------
ds_path = Path(args.data_path)/args.dataset/'datasets'

model_path = Path(args.data_path)/args.dataset/'models'

svds_path = Path(args.data_path)/args.dataset/args.model/'svds'/args.reduction

cvs_path = Path(args.data_path)/args.dataset/args.model/'corevectors'/args.reduction
cvs_name = 'cvs'

drill_path = Path(args.data_path)/args.dataset/args.model/'drillers'/args.reduction/args.analysis
drill_name = 'driller'

phs_path = Path(args.data_path)/args.dataset/args.model/'peepholes'/args.reduction/args.analysis
phs_name = 'phs'

tune_storage_path = Path(args.data_path)/args.dataset/args.model/'tuning'/args.reduction/args.analysis

hyper_params_file = phs_path/f'hyperparams.pickle'

results_path = Path.cwd()/'temp_results'

# detection metrics reported by the experiments, both of them the smaller the better
det_metrics = ['E-AURC', 'DetError']

# one report per detection metric
det_metrics_df_paths = {_m: results_path/f'{_m}_comparison.pickle' for _m in det_metrics}

# report used to skip the analyses already computed
saved_df_path = det_metrics_df_paths[det_metrics[0]]

# scores computed from the peepholes, one folder per reduction
scores_path = results_path/'scores'/args.dataset/args.model/args.reduction

# scores compared against, which do not depend on the reduction
others_scores_path = results_path/'scores'/args.dataset/args.model

#--------------------------------
# Runing
#--------------------------------

n_threads = 1
verbose = True
bs_base = 2**10
bs_atk_scale = 2**-4
tune_num_samples = 50
chunk_size = 5000 # divides parsed dataset into smaller files for efficiency

#--------------------------------
# Defs
#--------------------------------
atk_names = ['BIM', 'PGD', 'FAB-t', 'Square', 'APGD-ce', 'APGD-t']

def get_pos_loader():
    return f'{args.dataset}-test-{args.model}'

def get_neg_loaders_ood(ood_dss):
    return [f'{k}-test-{args.model}' for k in ood_dss.keys()]

def get_neg_loaders_aa(atks):
    return [f'{args.dataset}-test-{a}-{args.model}' for a in atks]

def get_neg_loaders_fit(ood_dss, atks):
    '''
    Negative test loaders mapped to the negative train loaders used to calibrate them, for the scores fitted against each negative loader.
    '''
    return {
            **{f'{k}-test-{args.model}': [f'{k}-val-{args.model}'] for k in ood_dss.keys()},
            **{f'{args.dataset}-test-{a}-{args.model}': [f'{args.dataset}-val-{a}-{args.model}'] for a in atks},
            }

def get_loaders(ood_dss):
    return (
        [f'{args.dataset}-train', f'{args.dataset}-val', f'{args.dataset}-test'] +
        [f'{k}-val' for k in ood_dss.keys()] +
        [f'{k}-test' for k in ood_dss.keys()]
    )

def get_inference_names(ood_dss):
    names = {
        f'{args.dataset}-train': [args.model],
        f'{args.dataset}-val': [args.model] + [f'{a}-{args.model}' for a in atk_names],
        f'{args.dataset}-test': [args.model] + [f'{a}-{args.model}' for a in atk_names],
    }
    for k in ood_dss.keys():
        names[f'{k}-val'] = [args.model]
        names[f'{k}-test'] = [args.model]
    return names

def get_transforms(ood_dss):
    return {k: TransformWrap(transform=transform, input_key='image') for k in get_loaders(ood_dss)}

cvs_names = {l: cvs_name for l in target_layers}
def get_atks(model, eps, steps):
    return {
        'BIM-'+args.model: myBIM(
            model = model,
            eps = eps,
            steps = steps,
            ),
        'PGD-'+args.model: myPGD(
            model = model,
            eps = eps,
            steps = steps,
            ),
        'FAB-t-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = eps,
                attacks_to_run = ['fab-t'],
                ),
        'Square-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = eps,
                attacks_to_run = ['square'],
                ),
        'APGD-ce-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = eps,
                attacks_to_run = ['apgd-ce'],
                ),
        'APGD-t-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = eps,
                attacks_to_run = ['apgd-t'],
                ),
        }
