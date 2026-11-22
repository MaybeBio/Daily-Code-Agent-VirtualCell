from scripts.utils import set_seed, create_id, set_arm, scgen_pred_path, scgen_result_dir, run_dir
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import sparse
import pandas as pd
import scanpy as sc
import numpy as np
import random
import scvi
import torch
import os


test_metric = pd.DataFrame(
    columns=['experiment_id', 'test', 'AR', 'variable_con',
             'con_percent','in_dist_group', 'metric',
             'value', 'seed', 'arm'])


def get_train_test_adata(args, adata):
    """Get the train and test data.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        
    Returns:
        train_adata (anndata.AnnData): Anndata object containing the train data.
        test_adata (anndata.AnnData): Anndata object containing the test data.
        unper_test_adata (anndata.AnnData): Anndata object containing the control test data.
        per_test_adata (anndata.AnnData): Anndata object containing the stimulated test data."""

    train_valid_adata = adata[~(
        (adata.obs[args.adata_label_cell].isin(args.test_data)) &
        (adata.obs.condition == args.adata_label_per))]
    
    set_seed(args.seed)
    valid_index = random.sample(range(train_valid_adata.shape[0]),
                                int(train_valid_adata.shape[0]/5))
    train_index = \
        [x for x in range(train_valid_adata.shape[0])
         if x not in valid_index]

    train_adata = train_valid_adata[train_index]
    valid_adata = train_valid_adata[valid_index]
    assert train_adata.shape[0] + valid_adata.shape[0] == \
        train_valid_adata.shape[0]

    # designed data: the model was trained on the 'train' cells, and only
    # 'test' cells of the held-out or in-distribution group are evaluated
    if getattr(args, 'designed_data', False):
        if 'split' not in adata.obs:
            raise ValueError('--designed_data: no obs["split"] in the data')
        train_adata = adata[adata.obs['split'] == 'train']
        adata = adata[adata.obs['split'] == 'test']

    if args.in_dist_group == '':
        unper_test_adata = adata[(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_unper))]
        per_test_adata = adata[(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_per))]
        
    elif args.in_dist_group != '':
        print('in_dist_group: '+args.in_dist_group)

        in_dist_adata = \
            adata[adata.obs[args.adata_label_cell] == args.in_dist_group]
            
        unper_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_unper)]
        per_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_per)]

    test_adata = per_test_adata.concatenate(unper_test_adata)
    return train_adata, test_adata, unper_test_adata, per_test_adata


def load_eval_adata(args, adata, model_name='Naive'):
    """Load the saved predictions of the given model for the test cell group.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        model_name (str, optional): Name of the model.
        
    Returns:
        anndata.AnnData: Anndata object containing the predicted data."""

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
        
    pred_adata = sc.read(scgen_pred_path(args))
    assert pred_adata.obs['condition'].unique()[0] == model_name
    assert pred_adata.shape == unper_test_adata.shape

    return pred_adata


def update_training_sample_probabilities(z, bins=100, smoothing_fac=0.0001): 
    """Updates the sampling probabilities for cells within a batch based on the maximum of
    latent variables' probablities.
    Adapted from MIT 6.S191 Lab 2 (DB-VAE), MIT License,
    © MIT Introduction to Deep Learning, http://introtodeeplearning.com
    (https://github.com/MITDeepLearning/introtodeeplearning/commit/9e6df30),
    via https://goodboychan.github.io/python/tensorflow/mit/2021/02/27/Debiasing.html
        
    Args:
        z (torch.Tensor): Latent variables.
        bins (int, optional): Number of bins to use for histogram. Defaults to 100.
        smoothing_fac (float, optional): Smoothing factor for histogram. Defaults to 0.0001."""

    training_sample_p = np.zeros(z.shape[0])
        
    # consider the distribution for each latent variable 
    for i in range(z.shape[1]):

        latent_distribution = z[:,i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data sample falls in 
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        # call the digitize function to find which bins in the latent distribution 
        #    every data sample falls in to
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        bin_idx = np.digitize(latent_distribution, bin_edges) 

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function 
        p = 1.0/(hist_smoothed_density[bin_idx-1])

        # normalize all probabilities
        p = p / np.sum(p)

        # update sampling probabilities by considering whether the newly
        # computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(p, training_sample_p)

    # final normalization
    w = training_sample_p/np.sum(training_sample_p)

    return w


def plot_umap_with_weight_annotation(adata, args, model, dimension='high_dim'):
    """Plot UMAP visualization of adata for the input model 
    annotated with resampling weights.
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        model (torch.nn.Module): Model used for prediction.
        dimension (str, optional): Dimension to use for UMAP visualization. Defaults to high_dim."""

    print('Tracked epoch: '+str(args.tracked_epoch))
    if not os.path.isdir('figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/'):
        os.makedirs('figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/')

    new_adata = adata.copy()

    with torch.no_grad():

        z = model.get_latent_representation(new_adata)
        # calculate the resampling weights
        w = update_training_sample_probabilities(z, args.bins, args.alpha)
        # normalize the weights to be between 0 and 1
        new_adata.obs['resampling_weight'] = w
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        new_adata.obs['normalized_resampling_weight'] = w

        # print statistics of the resampling weights
        print('Resampling weights statistics:')
        print('Min: ', np.min(new_adata.obs['normalized_resampling_weight']))
        print('Max: ', np.max(new_adata.obs['normalized_resampling_weight']))
        print('Mean: ', np.mean(new_adata.obs['normalized_resampling_weight']))
        print('Std: ', np.std(new_adata.obs['normalized_resampling_weight']))
                
        if dimension == 'latent':
            new_adata = sc.AnnData(X=z, obs=new_adata.obs.copy())
            
        # plot UMAP with weight annotation
        
        # Extract the UMAP coordinates and the values for the feature to color by
        # fix seed for reproducibility
        set_seed(args.seed)
        sc.pp.neighbors(new_adata, random_state=42)
        sc.tl.umap(new_adata, random_state=42)
        
        # Sort the .obs DataFrame based on the 'your_feature' column
        # sorted_indices = new_adata.obs.sort_values(by='normalized_resampling_weight').index

        # # Reorder the AnnData object based on these sorted indices
        # new_adata = new_adata[sorted_indices,:]

        sc.set_figure_params(dpi_save=600, format='pdf', figsize=(6,6))
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['lightgrey', 'darkred'])
        sc.pl.umap(
            new_adata, color="normalized_resampling_weight",
            wspace=0.4, 
            frameon=False,
            save='_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/'+ \
                args.id+'-w-annotated-epoch-'+args.tracked_epoch+'-umap.pdf',
                title='',
                cmap=red_cmap,
                show=False,
                size=18,
                alpha=1.0,              # transparency of points
                edgecolor="black",      # border color for points
                linewidth=0.1,           # border thickness)
        )

        # Extract UMAP coordinates
        umap = new_adata.obsm['X_umap']
        x, y = umap[:, 0], umap[:, 1]

        # Extract values
        values = new_adata.obs["normalized_resampling_weight"].values

        # Sort by values so low-weight (red) points are plotted last (on top)
        order = values.argsort()
        x, y, values = x[order], y[order], values[order]

        # Custom colormap
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['lightgrey', 'darkred'])
        plt.figure(figsize=(6, 6), dpi=600)
        plt.scatter(
            x, y,
            c=values,
            cmap=red_cmap,
            s=11,                 # marker size
            alpha=1.0,            # transparency
            edgecolors="black",   # border color
            linewidths=0.05       # border thickness
        )
        # show color bar
        cbar = plt.colorbar()
        cbar.set_label('Normalized Resampling Weight', fontsize=25)
        cbar.ax.tick_params(labelsize=20)
        plt.axis("off")
        plt.tight_layout()
        out_path = args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/' + \
                args.id+'-w-annotated-epoch-'+args.tracked_epoch+'-umap_matplotlib.png'
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()


        annotation_list = ['blood_atlas']
        annotation_label = ['']
        palette = {"Blood": "orchid", "Atlas": "cadetblue"}


        # extract UMAP coordinates
        umap = new_adata.obsm["X_umap"]
        labels = new_adata.obs[annotation_list[0]]  # assuming annotation_list has one column

        fig, ax = plt.subplots(figsize=(6, 6))
        for group in labels.unique():
            mask = labels == group
            ax.scatter(
                umap[mask, 0],
                umap[mask, 1],
                s=11,
                c=[palette[group]],  # color from palette
                edgecolor="black",
                linewidth=0.1,
                alpha=0.3 if group == "Blood" else 0.7,
                label=group
            )

        ax.legend(fontsize=35, markerscale=8)
        ax.set_title("")
        ax.set_xlabel("UMAP1", fontsize=25)
        ax.set_ylabel("UMAP2", fontsize=25)
        ax.set_frame_on = False
        # remove the border of the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.grid(False)
        # remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # adjust the size of legend
        ax.legend(fontsize=20, markerscale=3)


        # check if args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
        # dimension+'/' exists, if not create it
        if not os.path.isdir(args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/'):
            os.makedirs(args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/')

        plt.tight_layout()
        plt.savefig(
            args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/'+args.id+'-cellytpe-annotated-umap.png',
            dpi=600
        )
        plt.close()


        # print('umap saved in figures/umap_w_annotated/'+args.data+ \
        #     '/seed'+str(args.seed)+'/'+ \
        #     dimension+'/'+args.id+'-cellytpe-annotated-umap.png')

    return


def convert_matrix_to_tensor(X):
    """Convert a matrix to a tensor.
    
    Args:
        X (any array): Input matrix.
        
    Returns:
        X (torch.Tensor): Converted tensor."""
    
    if isinstance(X, (np.ndarray)):
        X = torch.from_numpy(X).float()
    else:
        X = torch.from_numpy(sparse.coo_matrix(X).toarray()).float()
    X = X.to(torch.float32)
    
    return X


def _match_clusters_to_true_labels(true_labels, cluster_labels):
    """Map each KMeans id to a unique GT label (Hungarian on contingency)."""
    true_labels = np.asarray(true_labels).astype(str)
    cluster_labels = np.asarray(cluster_labels).astype(str)
    true_cats = np.unique(true_labels)
    clust_cats = np.unique(cluster_labels)

    cont = np.zeros((len(clust_cats), len(true_cats)), dtype=np.int64)
    clust_index = {c: i for i, c in enumerate(clust_cats)}
    true_index = {t: i for i, t in enumerate(true_cats)}
    for t, c in zip(true_labels, cluster_labels):
        cont[clust_index[c], true_index[t]] += 1

    # Pad to square if needed so every cluster gets a distinct label when possible.
    n = max(cont.shape[0], cont.shape[1])
    cost = np.zeros((n, n), dtype=np.float64)
    cost[: cont.shape[0], : cont.shape[1]] = cont
    row_ind, col_ind = linear_sum_assignment(-cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(clust_cats) and c < len(true_cats):
            mapping[clust_cats[r]] = true_cats[c]
    for c in clust_cats:
        if c not in mapping:
            mapping[c] = true_cats[int(np.argmax(cont[clust_index[c]]))]
    return np.asarray([mapping[c] for c in cluster_labels], dtype=object)


def evaluate_delta_correlation(adata, args, DEG_sets):
    """Calculate the Pearson r, across genes, between the real response
    (mean stimulated - mean control) and the predicted response
    (mean prediction - mean control), with the real control cells of the
    held-out group as the reference.

    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        DEG_sets (dict): DEG set name ('up+down') -> {DEG count: indices of the DEGs}."""

    global test_metric
    # real response: stimulated - control
    unper_mean = np.mean(adata[adata.obs.condition == \
        args.adata_label_unper].X.toarray().astype(float), axis=0)
    per_mean = np.mean(adata[adata.obs.condition == \
        args.adata_label_per].X.toarray().astype(float), axis=0)
    per_diff = per_mean - unper_mean
    assert unper_mean.shape == per_mean.shape == per_diff.shape

    for model_name in args.model.split(','):
        set_arm(args, model_name)
        create_id(args)

        pred_mean = np.mean(adata[adata.obs.condition == \
            model_name].X.toarray().astype(float), axis=0)
        pred_diff = pred_mean - unper_mean
        diff_corr, p_value = pearsonr(per_diff, pred_diff)
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'Delta Pearson r', float(diff_corr),
             args.seed, args.arm]

        for DEG_set, DEG_idx_dic in DEG_sets.items():
            for DEG_count, DEG_idx in DEG_idx_dic.items():
                DEG_diff_corr, p_value = pearsonr(
                    per_diff[DEG_idx],
                    pred_diff[DEG_idx])
                test_metric.loc[test_metric.shape[0]] = \
                    [args.test_id, args.test_data[0], args.AR, args.variable_con,
                     args.con_percent, args.in_dist_group,
                     f'{DEG_count} {DEG_set} DEG Delta Pearson r',
                     float(DEG_diff_corr), args.seed, args.arm]

    return


def fit_distribution_space(unper_X, per_X, n_pcs=50, seed=0):
    """Fit the space used by the distribution metric: a PCA on the real control
    and perturbed cells of the held-out group.

    Args:
        unper_X (np.ndarray): Expression of the real control cells.
        per_X (np.ndarray): Expression of the real perturbed cells.
        n_pcs (int, optional): Number of principal components.
        seed (int, optional): Seed of the PCA.

    Returns:
        sklearn.decomposition.PCA: Fitted PCA."""
    X = np.concatenate([unper_X, per_X], axis=0)
    pca_model = PCA(n_components=n_pcs, random_state=seed)
    Z = pca_model.fit_transform(X)

    return pca_model


def _pairwise_sum(X, Y, fn, chunk=2048):
    """Sum of fn(Euclidean distance) over all pairs of rows of X and Y, in chunks.

    Args:
        X (torch.Tensor): First set of points (n x d).
        Y (torch.Tensor): Second set of points (m x d).
        fn (callable): Function applied to the distance matrix of each chunk.
        chunk (int, optional): Number of rows of X per chunk.

    Returns:
        float: Sum over all n x m pairs."""
    total = 0.0
    for start in range(0, X.shape[0], chunk):
        dist = torch.cdist(X[start:start+chunk], Y,
                           compute_mode='donot_use_mm_for_euclid_dist')
        total += fn(dist).sum().item()
    return total


def energy_distance(X, Y):
    """Energy distance between two sets of points, with Euclidean (not squared)
    distances: 2 E||x - y|| - E||x - x'|| - E||y - y'||. The within-set means
    leave out the pairs of a point with itself.

    Args:
        X (torch.Tensor): First set of points (n x d).
        Y (torch.Tensor): Second set of points (m x d).

    Returns:
        float: Energy distance."""
    n, m = X.shape[0], Y.shape[0]
    xy = _pairwise_sum(X, Y, lambda d: d) / (n * m)
    xx = _pairwise_sum(X, X, lambda d: d) / (n * (n - 1))
    yy = _pairwise_sum(Y, Y, lambda d: d) / (m * (m - 1))
    return 2 * xy - xx - yy


def evaluate_distribution_metrics(adata, args):
    """Compare the distribution of the predicted cells with the real perturbed
    cells of the held-out group: energy distance in a PCA space fit on the real
    control and perturbed cells (fit_distribution_space). All cells are used;
    the space is the same for every arm.

    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script."""

    global test_metric
    unper_X = adata[adata.obs.condition == \
        args.adata_label_unper].X.toarray().astype(float)
    per_X = adata[adata.obs.condition == \
        args.adata_label_per].X.toarray().astype(float)
    pca_model = fit_distribution_space(unper_X, per_X)
    per_Z = torch.from_numpy(pca_model.transform(per_X)).double()

    for model_name in args.model.split(','):
        set_arm(args, model_name)
        create_id(args)

        pred_X = adata[adata.obs.condition == \
            model_name].X.toarray().astype(float)
        pred_Z = torch.from_numpy(pca_model.transform(pred_X)).double()

        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'Energy Distance',
             float(energy_distance(pred_Z, per_Z)), args.seed, args.arm]

    return


def extract_up_down_DEG(test_adata, args, count=100):
    """Extract the top differentially expressed genes in both directions
    (perturbed vs control).

    Wilcoxon test over all genes, ranked by the absolute score, so genes that
    go up and genes that go down are both included.

    Args:
        test_adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        count (int, optional): Number of DEGs to extract.

    Returns:
        list: Indices of the DEGs."""
    sc.tl.rank_genes_groups(test_adata,
                            groupby="condition",
                            method="wilcoxon",
                            n_genes=test_adata.shape[1],
                            key_added="rank_genes_groups_all")
    ranking = test_adata.uns["rank_genes_groups_all"]
    names = np.array(ranking["names"][args.adata_label_per].tolist())
    scores = np.array(ranking["scores"][args.adata_label_per].tolist(), dtype=float)
    order = np.argsort(-np.abs(scores), kind="stable")
    diff_genes = names[order][:count].tolist()

    if args.data in ['species', 'lps-hpoly']:
        gene_list = test_adata.var.index.tolist()
    else:
        gene_list = test_adata.var["gene_symbol"].tolist()

    gene_pos = {gene: j for j, gene in enumerate(gene_list)}
    DEG_idx = [gene_pos[gene] for gene in diff_genes if gene in gene_pos]

    return DEG_idx


def create_test_id(args):
    """Create the test id.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script."""

    args.test_id = args.data+ \
        '-ood'+str(args.ood)+ \
        '-variable_con'+str(args.variable_con)

    if getattr(args, 'designed_data', False):
        args.test_id = args.test_id+'-designed'
        if getattr(args, 'design_abundant', None):
            args.test_id = args.test_id+'-main'+args.design_abundant
    if args.in_dist_group != '':
        args.test_id = args.test_id+'-indist'
    return


def test_scgen(args, adata=None, model_dic={}):
    """Test the model.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData, optional): Anndata object containing the data.
        model_dic (dict, optional): Dictionary containing the model for each method.
        """

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
    # designed data: the design (abundant group) names the results file
    if getattr(args, 'designed_data', False):
        args.design_abundant = adata.uns['design']['abundant']
    create_test_id(args)

    unper_test_adata.X = convert_matrix_to_tensor(unper_test_adata.X)
    per_test_adata.X = convert_matrix_to_tensor(per_test_adata.X)
    all_test_adata = per_test_adata.concatenate(unper_test_adata)
        
    assert set(unper_test_adata.obs[args.adata_label_cell]) == \
        set(per_test_adata.obs[args.adata_label_cell])

    models = args.model.split(',')
    for model_name in models:
        set_arm(args, model_name)
        create_id(args)
        pred_adata = load_eval_adata(args, adata, model_name)
        pred_adata.X = convert_matrix_to_tensor(pred_adata.X)
        all_test_adata = pred_adata.concatenate(all_test_adata)

        assert pred_adata.obs[args.adata_label_cell].tolist() == \
            unper_test_adata.obs[args.adata_label_cell].tolist()
        assert pred_adata.obs.condition.tolist() == \
            [model_name]*pred_adata.shape[0]
            
    # the combined test data must hold exactly the predicted arms of
    # args.model and the real control and perturbed cells
    assert set(all_test_adata.obs.condition) == \
        set(args.model.split(',')) | {args.adata_label_unper, args.adata_label_per}
        
    # extract DEG index: top 20 DEGs in both directions
    DEG_sets = {'up+down': {}}
    for count in [20]:
        DEG_sets['up+down'][count] = extract_up_down_DEG(
            per_test_adata.concatenate(unper_test_adata),
            args, count=count)

    # Pearson r between the real and predicted responses (perturbed - control)
    evaluate_delta_correlation(all_test_adata, args, DEG_sets)

    # energy distance between predicted and real perturbed cells
    evaluate_distribution_metrics(all_test_adata, args)

    # save results
    global test_metric
    if getattr(args, 'designed_data', False):
        # designed data: one file per eval job (held-out group, percent, seed, evaluated
        # group), so the jobs can run in parallel; merged per design afterwards
        group = args.in_dist_group or args.test_data[0]
        file = os.path.join(scgen_result_dir(args), 'jobs', args.test_id,
                            f'test{args.test_data[0]}-per{args.con_percent}-seed{args.seed}-{group}.csv')
        os.makedirs(os.path.dirname(file), exist_ok=True)
    else:
        file = scgen_result_dir(args)+ \
            "{:%Y%m%d}".format(datetime.now())+'-'+args.test_id+'.csv'
    # check if there is a file with the same name
    # metric values in full precision (9 significant digits)
    out = test_metric.assign(value=test_metric['value'].map('{:.8e}'.format))
    if getattr(args, 'designed_data', False):
        design = adata.uns['design']
        out = out.assign(abundant=design['abundant'],
                         design_file=os.path.basename(args.h5ad_adata_file))
    if not os.path.isfile(file):
        out.to_csv(file, index=False, header=True)
    else:
        out.to_csv(file, mode='a', index=False, header=False)

    return


def test_scvi(args, train_adata):
    """Test the scVI model.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        train_adata (anndata.AnnData): Anndata object containing the training data."""
    
    train_adata_copy = train_adata.copy()
    print('train_adata_copy shape: ', train_adata_copy.shape)

    # plot umap plots annotated by AR weights
    if args.plot_umap_annotated_with_w:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            # checkpoint <run_dir>/best or <run_dir>/epoch<N>
            path_to_model_folder = run_dir(args)+('best' if args.tracked_epoch == 'best'
                                                  else 'epoch'+str(args.tracked_epoch))
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            # plot_umap_with_weight_annotation(all_adata, args, model, 'latent')
            plot_umap_with_weight_annotation(train_adata_copy, args, model, 'high_dim')
            return
    
    # store resampling weights
    if args.store_resampling_weights:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            # checkpoint <run_dir>/best or <run_dir>/epoch<N>
            path_to_model_folder = run_dir(args)+('best' if args.tracked_epoch == 'best'
                                                  else 'epoch'+str(args.tracked_epoch))
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            with torch.no_grad():

                z = model.get_latent_representation(train_adata_copy)
                # calculate the resampling weights
                w = update_training_sample_probabilities(z, args.bins, args.alpha)

                # save the resampling weights in a csv file
                resampling_weights_df = pd.DataFrame({'resampling_weight': w})
                # assert number of rows in resampling_weights_df is equal to number of rows in train_adata_copy
                assert resampling_weights_df.shape[0] == train_adata_copy.shape[0]
                resampling_weights_df.to_csv(args.resampling_weight_path, index=False)
                print('saved resampling weights to: '+args.resampling_weight_path)

    # store weight diagnostics (normalized entropy and ESS)
    if args.store_weight_diagnostics:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            # checkpoint <run_dir>/best or <run_dir>/epoch<N>
            path_to_model_folder = run_dir(args)+('best' if args.tracked_epoch == 'best'
                                                  else 'epoch'+str(args.tracked_epoch))
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            with torch.no_grad():
                z = model.get_latent_representation(train_adata_copy)
                w = update_training_sample_probabilities(z, args.bins, args.alpha)

                n = len(w)
                # normalized Shannon entropy: 1 = uniform weights, 0 = all mass on one cell
                entropy = -np.sum(w * np.log(w + 1e-12))
                entropy_normalized = entropy / np.log(n)
                # effective sample size: how many cells are effectively being sampled
                ess = 1.0 / np.sum(w ** 2)
                ess_normalized = ess / n

                diagnostics_df = pd.DataFrame({
                    'entropy_normalized': [entropy_normalized],
                    'ess': [ess],
                    'ess_normalized': [ess_normalized],
                })
                diagnostics_df.to_csv(args.weight_diagnostics_path, index=False)
                print('entropy_normalized: ', entropy_normalized)
                print('ess_normalized: ', ess_normalized)
                print('saved weight diagnostics to: '+args.weight_diagnostics_path)
    return