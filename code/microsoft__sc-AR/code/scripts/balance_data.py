"""Balanced scTab training data for the scVI baselines (class balancing,
geometric sketching).

The 100k-cell corpus (blood cells, then atlas_count atlas cells) and its HVGs
are rebuilt as for Standard/AR (dataloader.read_and_preprocess_data_for_scvi_sctab;
the HVG file must exist and is checked). It is split into training and
validation cells exactly as for Standard/AR (10% blood validation cells), and
only the training cells are balanced, keeping 90,000 rows; the validation cells
are unchanged. Both keep raw counts. Writes into
<root>/data/<data>/balanced_data/<method>/<seed>/:
  bloodbase_<atlas>_atlas_seed<seed>_<hvg>HVGs_<method>_{train_adata.h5ad,valid_adata.h5ad,composition.csv}

usage (from the repository root):
  python scripts/balance_data.py --obs_class_label tissue --seed 42 --atlas_count 100 \
      --balancing_method class_balancing
"""
from utils import set_seed
from geosketch import gs
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import time
import os



def select_hvg(
    adata,
    hvg_count,
    original_train_adata_standard,
    original_train_adata_AR=None,
):
    """Select hvg from training data.
    
    Args:
        adata (AnnData): AnnData object containing the data
        hvg_count (int): number of hvg to select
        original_train_adata_standard (AnnData): AnnData object containing the original training data
        original_train_adata_AR (AnnData): AnnData object containing the original training data with AR
        
    Returns:
        adata (AnnData): AnnData object containing the data with the hvg selected
    """
    # select the genes from the original train_adata file
    gene_list = original_train_adata_standard.var_names.tolist()
    assert len(gene_list) == hvg_count
    adata = adata[:, gene_list]
    assert adata.shape[1] == original_train_adata_standard.shape[1]
    assert all(adata.var_names == original_train_adata_standard.var_names)
    # assert all(adata.var_names == original_train_adata_AR.var_names)

    # check if set of all cells in original_train_adata_AR is a subset of the set of all cells in the training data adata
    # assert set(original_train_adata_AR.obs_names.tolist()) <= set(adata.obs_names.tolist())

    return adata


def check_hvg_reproducible(
    adata,
    hvg_count,
    seed,
    original_train_adata_standard,
    path_to_original_train_adata_file,
):
    """Check that the HVG file can be reproduced from this corpus.

    Rebuilds the Standard/AR training cells (the corpus minus the 10% blood
    validation cells, drawn as in dataloader.read_and_preprocess_data_for_scvi_sctab),
    checks they are the cells of the HVG file, reruns seurat_v3 HVG selection on
    them and compares the genes. Also compares with the ARFalse file if it exists.

    Args:
        adata (AnnData): AnnData object containing the 100k corpus (blood cells first, all genes)
        hvg_count (int): number of hvg
        seed (int): seed for random number generator
        original_train_adata_standard (AnnData): AnnData object read from the HVG file
        path_to_original_train_adata_file (str): path to the HVG file
    """
    standard_train_adata, _ = split_data_as_standard_and_AR(adata, seed)

    if not np.array_equal(standard_train_adata.obs_names, original_train_adata_standard.obs_names):
        raise ValueError("Cells of the HVG file differ from the rebuilt Standard/AR training cells: "
                         + path_to_original_train_adata_file)

    sc.pp.highly_variable_genes(
        standard_train_adata,
        n_top_genes=hvg_count,
        subset=True,
        flavor='seurat_v3'
    )
    recomputed = set(standard_train_adata.var_names)
    saved = set(original_train_adata_standard.var_names)
    if recomputed != saved:
        raise ValueError(f"Recomputed HVGs differ from the HVG file: {len(recomputed & saved)} of "
                         f"{hvg_count} genes agree ({path_to_original_train_adata_file})")

    path_to_AR_false_file = path_to_original_train_adata_file.replace("_ARTrue_", "_ARFalse_")
    if os.path.exists(path_to_AR_false_file):
        AR_false_genes = sc.read_h5ad(path_to_AR_false_file, backed='r').var_names
        if not AR_false_genes.equals(original_train_adata_standard.var_names):
            raise ValueError("ARTrue and ARFalse HVG files have different genes")
    print("HVG check passed: same training cells and same", hvg_count, "genes")


def balance_data_class_balancing(
    adata,
    seed,
    obs_class_label,
):
    """Balance the data using class balancing.
    
    Args:
        adata (AnnData): AnnData object containing the data
        seed (int): seed for random number generator
        obs_class_label (str): obs class label to check
    """
    set_seed(seed)

    # get the unique values of the obs_class_label
    unique_values = adata.obs[obs_class_label].unique()
    sample_count_per_class = int(adata.shape[0] / len(unique_values))

    all_indices = []
    for value in unique_values:
        # get the indices of the cells with the obs_class_label equal to value
        mask = adata.obs[obs_class_label] == value
        indices = np.where(mask)[0]
        # sample from the indices with replacement
        # check if the number of cells in the class is greater than the sample_count_per_class, if so, sample from the class without replacement
        if len(indices) >= sample_count_per_class:
            sampled_indices = np.random.choice(indices, size=sample_count_per_class, replace=False)  
        else:
            sampled_indices = np.random.choice(indices, size=sample_count_per_class, replace=True)  
        
        all_indices.extend(sampled_indices)

    # check if balanced_adata.shape[0] != adata.shape[0], count the difference and randomly sample from the difference to make the number of cells equal to the original number of cells
    if len(all_indices) != adata.shape[0]:
        difference = adata.shape[0] - len(all_indices)
        sampled_indices = np.random.choice(adata.shape[0], size=difference, replace=True)
        all_indices.extend(sampled_indices)

    balanced_adata = adata[np.array(all_indices)].copy()
    assert balanced_adata.shape[0] == adata.shape[0]
    # assert there are the same number of classes as it was in the original adata
    assert len(balanced_adata.obs[obs_class_label].unique()) == len(unique_values)
    # Check the number of cells per class (may not be exactly equal due to rounding)
    for value in unique_values:
        actual_count = balanced_adata.obs[obs_class_label].value_counts()[value]
        print(f"Class {value}: {actual_count} cells (target: {sample_count_per_class})")
        # Note: Due to rounding and the adjustment step, counts may not be exactly equal
        # The assertion is commented out as it may fail due to integer division rounding
    return balanced_adata


def balance_data_geometric_sketching(
    adata,
    seed,
):
    """Balance the data using geometric sketching.

    The sketch has as many rows as the input (sampled with replacement) and uses
    floor(sqrt(N)) covering boxes.

    Args:
        adata (AnnData): AnnData object containing the data with PCA representation
        seed (int): seed for random number generator

    Returns:
        adata (AnnData): AnnData object containing the geometrically sketched data
        num_covering_boxes (int): number of covering boxes used
    """
    set_seed(seed)

    # derived params
    sketch_size = adata.shape[0]
    num_covering_boxes = int(np.sqrt(adata.shape[0]))

    print("Starting geometric sketching")
    start = time.time()

    sketch_index = gs(
        adata.obsm['X_pca'],
        sketch_size,
        seed=seed,
        k=num_covering_boxes,
        replace=True,
    )

    end = time.time()
    print("Geometric sketching complete")
    print("Time Taken", end - start)

    # Create balanced AnnData object with only the sketched cells
    balanced_adata = adata[sketch_index].copy()
    balanced_adata.obs_names = [
        f"{name}_gs{i}" for i, name in enumerate(balanced_adata.obs_names)
    ]
    print(f"Balanced data shape: {balanced_adata.shape} (original: {adata.shape})")
    
    return balanced_adata, num_covering_boxes


def split_data(
    adata,
    seed,
):
    """Split the data into training and validation sets.
    
    Args:
        adata (AnnData): AnnData object containing the data
        seed (int): seed for random number generator

    Returns:
        train_adata (AnnData): AnnData object containing the training data
        valid_adata (AnnData): AnnData object containing the validation data
    """
    set_seed(seed)

    train_ratio = 0.9
    train_index = np.random.choice(adata.shape[0], int(train_ratio*adata.shape[0]), replace=False)
    train_index = np.array(train_index)
    valid_index = [x for x in range(adata.shape[0]) if x not in train_index]
    valid_index = np.array(valid_index)
    valid_adata = adata[valid_index]
    train_adata = adata[train_index]
    assert train_adata.shape[0] + valid_adata.shape[0] == adata.shape[0]
    return train_adata, valid_adata


def split_data_as_standard_and_AR(
    adata,
    seed,
):
    """Split the corpus into training and validation sets exactly as for Standard/AR.

    The validation set is 10% of all training cells drawn from the blood cells only,
    with the same calls as dataloader.read_and_preprocess_data_for_scvi_sctab.
    The training set is all other cells (remaining blood cells, then all atlas cells).

    Args:
        adata (AnnData): AnnData object containing the 100k corpus (blood cells first)
        seed (int): seed for random number generator

    Returns:
        train_adata (AnnData): AnnData object containing the training data
        valid_adata (AnnData): AnnData object containing the validation data
    """
    all_train_cells = 100000
    blood_count = (adata.obs['blood_atlas'] == 'Blood').sum()
    # blood cells come first in the corpus, as in the Standard/AR concatenation
    assert (adata.obs['blood_atlas'].values[:blood_count] == 'Blood').all()
    set_seed(seed)
    valid_index = np.random.choice(blood_count, int(0.1*all_train_cells), replace=False)
    train_index = np.setdiff1d(np.arange(adata.shape[0]), valid_index)
    train_adata = adata[train_index].copy()
    valid_adata = adata[valid_index].copy()
    assert train_adata.shape[0] + valid_adata.shape[0] == all_train_cells
    assert (valid_adata.obs['blood_atlas'] == 'Blood').all()
    return train_adata, valid_adata


def cell_composition(
    obs_by_split,
    obs_keys=('blood_atlas', 'tissue_general', 'tissue', 'cell_type'),
):
    """Count rows and unique cells per label for each data split.

    Args:
        obs_by_split (dict): split name -> obs DataFrame with a 'source_cell' column
        obs_keys (tuple): obs fields to count

    Returns:
        composition (DataFrame): one row per split, obs field and label with
            n_rows, n_unique_cells and fraction_rows
    """
    composition = []
    for split, obs in obs_by_split.items():
        for key in obs_keys:
            grouped = obs.groupby(key, observed=True)['source_cell']
            counts = pd.DataFrame({
                'n_rows': grouped.size(),
                'n_unique_cells': grouped.nunique(),
            })
            counts['fraction_rows'] = counts['n_rows'] / obs.shape[0]
            counts = counts.rename_axis('label').reset_index()
            counts.insert(0, 'obs_key', key)
            counts.insert(0, 'split', split)
            composition.append(counts)
    return pd.concat(composition, ignore_index=True)


def read_and_preprocess_data_for_scvi_sctab(
    atlas_adata, 
    blood_adata,
    atlas_count,
    hvg_count,
    seed,
    root,
    data
):
    """Read and preprocess data for scvi.
    
    Args:
        atlas_adata (AnnData): AnnData object containing the atlas data
        blood_adata (AnnData): AnnData object containing the blood data
        atlas_count (int): number of atlas cells
        hvg_count (int): number of hvg
        seed (int): seed for random number generator
        root (str): root directory

    Returns:
        train_adata (AnnData): AnnData object containing the training data
    """
    all_train_cells = 100000

    # read atlas data
    atlas_adata = sc.read_h5ad(atlas_adata)
    # read blood data
    blood_adata = sc.read_h5ad(blood_adata)

    # randomely select args.atlas_count number of cells from the atlas data
    set_seed(seed)
    index = np.random.choice(atlas_adata.shape[0], atlas_count, replace=False)
    train_adata_altas = atlas_adata[index]
            
    set_seed(seed)
    blood_count = all_train_cells - atlas_count
    index = np.random.choice(blood_adata.shape[0], blood_count, replace=False)
    train_adata_blood = blood_adata[index]
    assert train_adata_blood.shape[0]+train_adata_altas.shape[0] == all_train_cells
    
    # add a new field to the obs of the train_adata_blood
    train_adata_blood.obs['blood_atlas'] = 'Blood'
    train_adata_altas.obs['blood_atlas'] = 'Atlas'

    train_adata_blood.X = train_adata_blood.X.astype('float64')
    train_adata_altas.X = train_adata_altas.X.astype('float64')

    train_adata = train_adata_blood.concatenate(train_adata_altas)
    assert train_adata.shape[0] == all_train_cells

    train_adata.X = train_adata.X.astype('float64')
    assert train_adata.obs[train_adata.obs['blood_atlas'] == 'Atlas'].shape[0] == atlas_count

    # select the same hvg from training data
    path_to_original_train_adata_file = root+"/data/"+data+"/bloodbase_"+ \
        str(atlas_count)+'_atlas'+\
        "_seed"+str(seed)+"_ARTrue"+\
        "_train_adata_"+str(hvg_count)+"_"+str(hvg_count)+"HVGs.h5ad"
    original_train_adata_standard = sc.read_h5ad(path_to_original_train_adata_file)

    check_hvg_reproducible(
        train_adata,
        hvg_count,
        seed,
        original_train_adata_standard,
        path_to_original_train_adata_file,
    )

    train_adata = select_hvg(
        train_adata,
        hvg_count,
        original_train_adata_standard
    )

    return train_adata


def create_PCA_representation_for_geometric_sketching(
    adata,
    seed,
):
    """Create PCA representation for geometric sketching.
    
    Args:
        adata (AnnData): AnnData object containing the data (expected to be raw counts)
        seed (int): seed for random number generator

    Returns:
        adata (AnnData): AnnData object containing the data (X unchanged, raw counts)
            with the PCA representation in obsm['X_pca']
    """
    set_seed(seed)

    # Normalize and log-transform a copy for the PCA only, so that X keeps the raw
    # counts used to train scVI (as for Standard/AR and class balancing)
    pca_adata = adata.copy()
    sc.pp.normalize_total(pca_adata, target_sum=1e4)
    sc.pp.log1p(pca_adata)
    sc.pp.pca(pca_adata,
              svd_solver="arpack",
              random_state=seed)
    adata.obsm['X_pca'] = pca_adata.obsm['X_pca']
    # check if the PCA representation is created
    assert adata.obsm['X_pca'] is not None
    assert adata.obsm['X_pca'].shape[0] == adata.shape[0]
    return adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, nargs='?',
                        default="sctab",
                        help="data: sctab")
    parser.add_argument('--anndata_blood_file', default="data/sctab/BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad",
                        help="path to the anndata file")
    parser.add_argument('--anndata_atlas_file', default="data/sctab/BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad",
                        help="path to the anndata file")
    parser.add_argument("--obs_class_label",
                        nargs="+",
                        type=str,
                        required=True,
                        help="obs class label to check")
    parser.add_argument("--training_data_count", type=int, nargs='?',
                        default=100000,
                        help="training data count")
    parser.add_argument("--seed", type=int, nargs='?',
                        default=42,
                        help="seed for random number generator")
    parser.add_argument("--atlas_count", type=int, nargs='?',
                        default=0,
                        help="atlas count")
    parser.add_argument("--hvg_count", type=int, nargs='?',
                        default=2000,
                        help="hvg count")
    parser.add_argument("--balancing_method", type=str, nargs='?',
                        default="class_balancing",
                        help="balancing method: class_balancing, geometric_sketching")
    parser.add_argument("--root", type=str, nargs='?',
                        default="./",
                        help="root directory (repository root)")

    # parse and preprocess args
    args = parser.parse_args()
    args.anndata_atlas_file = args.root+args.anndata_atlas_file
    args.anndata_blood_file = args.root+args.anndata_blood_file

    # check if len of obs_class_label is 1
    if len(args.obs_class_label) == 1:
        args.obs_class_label = args.obs_class_label[0]

    ## read and preprocess data
    # create original atlas and blood combination of training data
    adata = read_and_preprocess_data_for_scvi_sctab(
        args.anndata_atlas_file,
        args.anndata_blood_file,
        args.atlas_count,
        args.hvg_count,
        args.seed,
        args.root,
        args.data,
    )

    # split the data into training and validation sets exactly as for Standard/AR
    # (10% blood validation cells) and balance the training data only
    adata, valid_adata = split_data_as_standard_and_AR(
        adata,
        args.seed,
    )
    # name of the original cell of every row (balancing repeats or renames rows)
    adata.obs['source_cell'] = adata.obs_names
    valid_adata.obs['source_cell'] = valid_adata.obs_names
    standard_train_obs = adata.obs.copy()

    # balance the data
    if args.balancing_method == "geometric_sketching":
        start_time = time.time()
        # Create PCA representation for geometric sketching (required for this method)
        adata = create_PCA_representation_for_geometric_sketching(
            adata,
            args.seed,
        )
        adata, args.num_covering_boxes = balance_data_geometric_sketching(
            adata,
            args.seed,
        )
        end_time = time.time()
        print("Time taken to create PCA representation and balance data for geometric sketching: ", end_time - start_time, "seconds")
    elif args.balancing_method == "class_balancing":
        adata = balance_data_class_balancing(
            adata,
            args.seed,
            args.obs_class_label,
        )
    else:
        raise ValueError("Invalid balancing method")


    # the balanced training data keeps the Standard/AR training size, the validation
    # data are the unchanged Standard/AR validation cells, both with raw counts
    train_adata = adata
    assert train_adata.shape[0] == 90000
    assert valid_adata.shape[0] == 10000
    assert (valid_adata.obs['blood_atlas'] == 'Blood').all()
    for name, split_adata in [("train", train_adata), ("valid", valid_adata)]:
        X = split_adata.X.data if sp.issparse(split_adata.X) else np.asarray(split_adata.X)
        assert np.all(np.mod(X, 1) == 0), name + " data are not raw counts"

    # composition of the training data before and after balancing, and of the validation data
    composition = cell_composition({
        'standard_train': standard_train_obs,
        'balanced_train': train_adata.obs,
        'valid': valid_adata.obs,
    })
    # balancing settings stored with the balanced training data
    train_adata.uns['balancing'] = {
        'method': args.balancing_method,
        'seed': args.seed,
        'atlas_count': args.atlas_count,
        'n_rows': train_adata.shape[0],
        'n_unique_cells': train_adata.obs['source_cell'].nunique(),
    }
    if args.balancing_method == "geometric_sketching":
        train_adata.uns['balancing']['num_covering_boxes'] = int(args.num_covering_boxes)
    print("Balancing summary:", train_adata.uns['balancing'])

    # check if args.root+"/data/$data/balanced_data/" exists, if not create it
    output_path = args.root+"/data/"+args.data+"/balanced_data/"+args.balancing_method+"/"+str(args.seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save the balanced train and validation data
    path_to_train_adata_file = output_path+"/bloodbase_"+ \
        str(args.atlas_count)+'_atlas'+\
        "_seed"+str(args.seed)+\
        "_"+str(args.hvg_count)+"HVGs"+\
        "_"+args.balancing_method

    composition.to_csv(path_to_train_adata_file+"_composition.csv", index=False)
    path_to_train_adata_file = path_to_train_adata_file+"_train_adata.h5ad"
    train_adata.write(path_to_train_adata_file)

    path_to_valid_adata_file = output_path+"/bloodbase_"+ \
        str(args.atlas_count)+'_atlas'+\
        "_seed"+str(args.seed)+\
        "_"+str(args.hvg_count)+"HVGs"+\
        "_"+args.balancing_method
    path_to_valid_adata_file = path_to_valid_adata_file+"_valid_adata.h5ad"
    valid_adata.write(path_to_valid_adata_file)
