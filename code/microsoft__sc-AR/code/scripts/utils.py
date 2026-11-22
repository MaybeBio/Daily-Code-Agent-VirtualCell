import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
import hashlib
import json
import random
import torch
import os


def set_seed(seed):
    """Set seed for reproducibility.
    
    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


def str2bool(v):
    """Convert string to boolean.
    
    Args:
        v (str): string to convert to boolean"""

    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return
    
    
# --w_update values and the aggregation rule in scripts/ar_aggregation_variants.py
W_UPDATE_METHODS = {'max_w': 'max', 'uniform': 'uniform', 'mean_w': 'mean',
                    'geomean_w': 'geomean', 'knn_w': 'knn'}

# scGen arms: Standard, Uniform, and AR with the max, mean, geomean and kNN
# rules, all with weights from the posterior mean
ARMS = ['Naive', 'Uniform', 'AR-mu', 'AR-mean-mu', 'AR-geomean-mu', 'AR-knn-mu']

# k of the AR-kNN rule (default of calculate_resampling_weights_with_method)
KNN_K = 30


def set_arm(args, model_name):
    """Set args.AR and args.w_update for a scGen arm name.

    Args:
        args (argparse): input arguments
        model_name (str): one of ARMS: 'Naive' (standard), 'Uniform', or
            'AR[-<rule>]-mu': AR with the max rule or another aggregation rule
            (<rule> in mean, geomean, knn), weights from the posterior mean"""

    if model_name not in ARMS:
        raise ValueError('Unknown model name: '+model_name)
    args.AR, args.w_update = True, 'max_w'
    parts = model_name.split('-')[:-1]
    if model_name == 'Naive':
        args.AR = False
    elif model_name == 'Uniform':
        args.w_update = 'uniform'
    elif len(parts) == 2:
        args.w_update = parts[1]+'_w'
    args.arm = model_name

    return


def arm_name(args):
    """Arm name of the run given by args (inverse of set_arm).

    Args:
        args (argparse): input arguments

    Returns:
        str: 'Naive', 'Uniform' or 'AR[-<rule>]-mu'"""

    if not args.AR:
        return 'Naive'
    if args.w_update == 'uniform':
        return 'Uniform'
    name = 'AR'
    if args.w_update != 'max_w':
        name += '-'+W_UPDATE_METHODS[args.w_update]
    if uses_latent_mean(args):
        name += '-mu'
    return name


def uses_latent_mean(args):
    """True for a scGen AR arm whose weights come from the posterior mean (all
    AR arms except Uniform). scVI arms keep their names ('Naive', 'AR').

    Args:
        args (argparse): input arguments"""

    return (getattr(args, 'model_name', 'scgen') == 'scgen' and args.AR
            and getattr(args, 'w_update', 'max_w') != 'uniform')


def create_id(args):
    """Create id for the experiment based on the input arguments.
    
    Args:
        args (argparse): input arguments"""
    
    if args.model_name == 'scgen':
        test_data = ''
        if args.ood:
            test_data = '-'.join(args.test_data)

        args.id = args.data+ \
            '-test'+test_data + \
            '-AR'+str(args.AR)+ \
            '-ood'+str(args.ood)+ \
            '-varcon'+str(args.variable_con)+ \
            '-per'+str(args.con_percent)+ \
            '-lr'+str(args.lr)+ \
            '-wd'+str(args.weight_decay)+ \
            '-bs'+str(args.batch_size)+ \
            '-ldim'+str(args.latent_dim)+ \
            '-epoch'+str(args.num_epoch)+ \
            '-seed'+str(args.seed)

    if args.model_name == 'scvi':
        args.id = 'bloodbase-'+str(args.atlas_count)+'atlas'+ \
            '-AR'+str(args.AR)[0]+ \
            '-lr'+str(args.lr)+ \
            '-wd'+str(args.weight_decay)+ \
            '-bs'+str(args.batch_size)+ \
            '-ldim'+str(args.latent_dim)+ \
            '-bins'+str(args.bins)+ \
            '-alpha'+str(args.alpha)+ \
            '-epoch'+str(args.num_epoch)

        if args.balanced_data:
            args.id = args.id + \
                '-'+str(args.balancing_method)

        args.id = args.id+'-scvi'+ \
            '-hvg'+str(args.hvg_count)+ \
            '-s'+str(args.seed)

    # AR arm suffixes (both models): aggregation rule other than max, and
    # weights from the posterior mean
    if args.AR and getattr(args, 'w_update', 'max_w') != 'max_w':
        args.id = args.id+'-w'+W_UPDATE_METHODS[args.w_update]
    if uses_latent_mean(args):
        args.id = args.id+'-latmu'

    return


def create_optimizers(model, args):
    """Create and return optimizer.

    Args_
        model (nn.Module): Neural network model to train.
        optimizer (torch.optim): the optimizer function to be used during
        training.
        args (_type_): _description_

    Returns_
        _type_: _description_

    """
    if args.optimizer.lower() == 'sgd':
        print("SGD Initialized")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            nesterov=False)

    elif args.optimizer.lower() == 'adam':
        print("Adam Initialized")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999))

    elif args.optimizer.lower() == 'radam':
        print("RAdam Initialized")
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,)

    elif args.optimizer.lower() == 'adamw':
        print("AdamW Initialized")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,)

    return optimizer




def run_settings(args):
    """Settings that define a trained model. They are hashed into the run
    directory name and written to config.json. Only values the training uses
    are included: bins and alpha only for the histogram-based AR rules, k only
    for AR-kNN; scVI weight decay is the scvi-tools default (set in main.py).

    Args:
        args (argparse): input arguments

    Returns:
        dict: settings of the run"""

    if args.model_name == 'scgen':
        settings = {
            'model_name': args.model_name, 'data': args.data,
            'data_file': os.path.basename(args.h5ad_adata_file),
            'train_data': sorted(args.train_data), 'test_data': sorted(args.test_data),
            'arm': arm_name(args), 'latent_dim': args.latent_dim,
            'num_epoch': args.num_epoch, 'optimizer': args.optimizer.lower(),
            'batch_size': args.batch_size, 'lr': args.lr,
            'weight_decay': args.weight_decay, 'seed': args.seed}
        uses_histogram = args.AR and args.w_update in ('max_w', 'mean_w', 'geomean_w')
        if args.AR and args.w_update == 'knn_w':
            settings['knn_k'] = KNN_K

    elif args.model_name == 'scvi':
        settings = {
            'model_name': args.model_name, 'data': args.data,
            'atlas_count': args.atlas_count,
            'balancing_method': args.balancing_method if args.balanced_data else None,
            'arm': arm_name(args), 'lr': args.lr, 'weight_decay': args.weight_decay,
            'batch_size': args.batch_size, 'latent_dim': args.latent_dim,
            'num_epoch': args.num_epoch, 'hvg_count': args.hvg_count, 'seed': args.seed}
        uses_histogram = args.AR

    if uses_histogram:
        settings['bins'] = args.bins
        settings['alpha'] = args.alpha

    return settings


def run_hash(args):
    """Short hash of run_settings(args).

    Args:
        args (argparse): input arguments

    Returns:
        str: first 8 hex characters of the SHA-1 of the run settings"""

    key_str = json.dumps(run_settings(args), sort_keys=True, default=str)

    return hashlib.sha1(key_str.encode()).hexdigest()[:8]


def run_dir(args):
    """Directory holding everything of one run: checkpoints, config.json,
    loss curves and predictions.

    Args:
        args (argparse): input arguments

    Returns:
        str: <out_path>/<model_name>/<data>/<hash>/"""

    return os.path.join(args.out_path, args.model_name, args.data, run_hash(args)) + '/'


def scgen_ckpt_path(args, tag):
    """Path of a scGen checkpoint (a directory holding model.pt).

    Args:
        args (argparse): input arguments
        tag (str): 'last', 'best' or 'epoch<N>'

    Returns:
        str: checkpoint path"""

    return run_dir(args)+tag+'/'


def scgen_pred_path(args):
    """Path of the saved scGen predictions (AnnData).

    Args:
        args (argparse): input arguments

    Returns:
        str: prediction file path"""

    suffix = '' if args.in_dist_group == '' else '-'+args.in_dist_group
    return run_dir(args)+'pred-adata'+suffix+'.h5ad'


def scgen_loss_path(args, split):
    """Path of the per-epoch loss curve of a scGen run.

    Args:
        args (argparse): input arguments
        split (str): 'train' or 'valid'

    Returns:
        str: csv path"""

    return run_dir(args)+split+'_loss.csv'


def scgen_result_dir(args):
    """Directory for scGen test-metric csv files.

    Args:
        args (argparse): input arguments

    Returns:
        str: directory path"""

    return os.path.join(args.root, 'result', 'test', args.data) + '/'


def write_run_config(args):
    """Write config.json with the run settings into the run directory.

    Args:
        args (argparse): input arguments"""

    with open(run_dir(args)+'config.json', 'w') as f:
        json.dump(run_settings(args), f, indent=2)

    return


def update_run_config(args, **fields):
    """Add fields (e.g. best_epoch, training time) to config.json of a run.

    Args:
        args (argparse): input arguments
        **fields: entries to add or overwrite"""

    path = run_dir(args)+'config.json'
    with open(path) as f:
        config = json.load(f)
    config.update(fields)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

    return


def write_runs_index(out_path, model_name, data):
    """Write runs.csv (one row per run: hash + config.json) for all runs in
    <out_path>/<model_name>/<data>/.

    Args:
        out_path (str): base directory of the saved models
        model_name (str): 'scgen' or 'scvi'
        data (str): data name"""

    base = os.path.join(out_path, model_name, data)
    rows = []
    for h in sorted(os.listdir(base)):
        path = os.path.join(base, h, 'config.json')
        if os.path.isfile(path):
            with open(path) as f:
                config = json.load(f)
            rows.append({'hash': h, **{k: ':'.join(map(str, v)) if isinstance(v, list) else v
                                       for k, v in config.items()}})
    pd.DataFrame(rows).to_csv(os.path.join(base, 'runs.csv'), index=False)

    return


def create_directory(args):
    """Create the directory to store the results.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script."""

    if args.model_name == 'scgen':
        if args.train:
            if os.path.exists(scgen_ckpt_path(args, 'best')):
                raise FileExistsError(
                    'A trained model already exists in '+run_dir(args)+
                    '. Delete that directory to retrain this run.')
            os.makedirs(run_dir(args), exist_ok=True)
        if args.test:
            os.makedirs(scgen_result_dir(args), exist_ok=True)
        return

    if args.train:
        if os.path.exists(os.path.join(run_dir(args), 'best')):
            raise FileExistsError(
                'A trained model already exists in '+run_dir(args)+
                '. Delete that directory to retrain this run.')
        os.makedirs(run_dir(args), exist_ok=True)

    if args.test:
        if not os.path.isdir(args.root+'/result/test/'+args.data):
            os.makedirs(args.root+'/result/test/'+args.data)

    return


def plot_sample_count(adata, args, path):
    
    # print adata with count number of each cell group a and condition
    print(adata.obs.groupby(['condition', args.adata_label_cell]).size())
    
    count_df = pd.DataFrame(columns=['Cell group', 'Condition', 'Number of cells'])
    per = adata[adata.obs['condition'] == args.adata_label_per]
    unper = adata[adata.obs['condition'] == args.adata_label_unper]
    y_per = list(set(per.obs[args.adata_label_cell].tolist()))
    y_unper = list(set(unper.obs[args.adata_label_cell].tolist()))
    # replace 'Enterocyte.Progenitor' with 'Ent.Progenitor'
    y_per = [x if x != 'Enterocyte.Progenitor' else 'Ent.Progenitor' for x in y_per]
    y_unper = [x if x != 'Enterocyte.Progenitor' else 'Ent.Progenitor' for x in y_unper]
    
    print(y_per)
    print(y_unper)
    
    # sort the cell groups
    y_per.sort()
    y_unper.sort()

    x_per = list()
    for group in y_per:
        original_group = group
        if group =='Ent.Progenitor':
            original_group = 'Enterocyte.Progenitor'
        x_per.append(per.obs[args.adata_label_cell].tolist().count(original_group))
        count_df.loc[count_df.shape[0]] = {'Cell group': group, 'Condition': 'Perturbed', 'Number of cells': per.obs[args.adata_label_cell].tolist().count(original_group)}
    x_unper = list()
    for group in y_unper:
        original_group = group
        if group =='Ent.Progenitor':
            original_group = 'Enterocyte.Progenitor'
        x_unper.append(unper.obs[args.adata_label_cell].tolist().count(original_group))
        count_df.loc[count_df.shape[0]] = {'Cell group': group, 'Condition': 'Control', 'Number of cells': unper.obs[args.adata_label_cell].tolist().count(original_group)}

    colors = {'Perturbed':'darkorange',
              'Control':'dodgerblue'}

    # create stacked barplot for the number of cells in each cell group with seaborn
    plt.clf()
    fig, ax = plt.subplots()
    assert y_per == y_unper
    cell_groups = tuple(y_unper)
    print(cell_groups)
    counts = {
        "Control": np.array(x_unper),
        "Perturbed": np.array(x_per),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(cell_groups))

    for boolean, count in counts.items():
        p = ax.bar(cell_groups, count, width, label=boolean, bottom=bottom, color=colors[boolean])
        bottom += count
        
    # adjust font size
    ax.set_ylabel('Number of cells', fontsize = 17)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=13, labelrotation=40)
    ax.tick_params(axis='y', labelsize=12)
    # set font size of legend

    # locate legend outside of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=13)
    
    plt.savefig(path+args.data+'-sample-count.png', dpi=600, bbox_inches="tight")
    return


def plot_sample_count_scvi(adata, args, path):
    
    # print adata with count number of each cell group a and condition
    print(adata.obs.groupby(['cell_type']).size())
    count_df = adata.obs.groupby(['cell_type']).size()
    
    count_df.columns = ['Cell group', 'Number of cells']
    # plot a barplot for the number of cells in each cell group
    plt.clf()
    fig, ax = plt.subplots()
    cell_groups = tuple(count_df.index)
    counts = count_df.values
    width = 0.5
    p = ax.bar(cell_groups, counts, width, color='dodgerblue')
        
    # adjust font size
    ax.set_ylabel('Number of cells', fontsize = 17)
    ax.set_xlabel('Cell group', fontsize = 17)
    # ax.tick_params(axis='x', labelsize=12, labelrotation=20)
    # ax.tick_params(axis='y', labelsize=16)
    # remove axis labels
    ax.set_xticklabels([])
    ax.legend()
    plt.savefig(path+args.data+'-sample-count.png', dpi=600, bbox_inches="tight")
    return


def quality_control_adata(adata, args):
    """Performs quality control on AnnData object and outputs relevant plots and statistics
    
    Args:
        adata (AnnData): AnnData object containing relevant count information with celltype
            and condition observations.
        args (argparse.ArgumentParser): It is a container for argument specifications.
    """
    print(args.data+' quality control')
    if args.data != 'lps-hpoly':
        adata.X = adata.X.toarray()

    address = 'figures/qq/'+args.data+'/'
    if not os.path.isdir(address):
        os.makedirs(address)
        
    if args.data == 'species':
        adata.obs[args.adata_label_cell] = adata.obs[args.adata_label_cell].str.capitalize()

    if args.model_name == 'scvi':
        plot_sample_count_scvi(adata, args, 'figures/qq/'+args.data+'/')
        return
    else:
        plot_sample_count(adata, args, 'figures/qq/'+args.data+'/')

    adata.obs['condition'] = adata.obs['condition'].replace(args.adata_label_unper, 'Control')
    adata.obs['condition'] = adata.obs['condition'].replace(args.adata_label_per, 'Perturbed')

    adata.obs[args.adata_label_cell] = adata.obs[args.adata_label_cell].replace('Enterocyte.Progenitor', 'Enterocyte.P')
    set_seed(args.seed)
    sc.pp.neighbors(adata)
    sc.set_figure_params(dpi=600, format='png', fontsize=13)
    
    palette = {'Perturbed':'darkorange',
              'Control':'dodgerblue'}
    
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=['condition'],
        wspace=0.4, frameon=False,
        save='-'+args.data+'-conditions',
        title='',
        legend_fontsize=16,
        palette=palette)
    sc.pl.umap(
        adata,
        color=[args.adata_label_cell],
        wspace=0.4, frameon=False,
        save='-'+args.data+'-celltypes',
        title='',
        legend_fontsize=16)
    
    return
