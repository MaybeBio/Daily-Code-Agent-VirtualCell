from scripts.scgen_AR.scgen._scgen import SCGEN
from scripts.utils import set_seed, create_id, set_arm, scgen_ckpt_path, scgen_pred_path
import random


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
    # 'test' cells of the held-out or in-distribution group are predicted
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
        in_dist_adata = \
            adata[adata.obs[args.adata_label_cell] == args.in_dist_group]
            
        unper_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_unper)]
        per_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_per)]

    test_adata = per_test_adata.concatenate(unper_test_adata)
    return train_adata, test_adata, unper_test_adata, per_test_adata


def predict_and_save_anndata_scgen(args, adata, model_name='Naive'):
    """Predict the perturbed gene expression values for the given test cell group
    and save it.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        model_name (str, optional): Name of the model."""

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
    
    model_path = scgen_ckpt_path(args, 'best')
    model = SCGEN.load(
        model_path,
        train_adata.copy(),
        accelerator="gpu" if args.gpu else "cpu",
        device='auto')
    model.is_trained = True
    print('Loaded model from:', model_path)

    # the held-out group is not a category of the model's setup when none of its
    # cells were in training: label its control cells with a training group (the
    # label is not an input of the scGen network); restored after predicting
    if getattr(args, 'designed_data', False) and args.in_dist_group == '':
        unper_test_adata.obs[args.adata_label_cell] = \
            train_adata.obs[args.adata_label_cell].unique()[0]

    set_seed(args.seed)
    pred_adata, delta = model.predict(
        ctrl_key=args.adata_label_unper,
        stim_key=args.adata_label_per,
        adata_to_predict=unper_test_adata
    )

    if args.in_dist_group == '':
        test_cell = args.test_data[0]
    elif args.in_dist_group != '':
        test_cell = args.in_dist_group

    unper_test_adata.obs[args.adata_label_cell] = test_cell
    pred_adata.obs[args.adata_label_cell] = test_cell
    pred_adata.obs.condition = model_name
    assert pred_adata.shape == unper_test_adata.shape

    pred_path = scgen_pred_path(args)
    pred_adata.write(pred_path)
    print("Saved prediction to:", pred_path)

    return


def predict(args, adata=None):
    """Make prediction.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData, optional): Anndata object containing the data."""

    # scGen predictions are saved inside each run directory
    models = args.model.split(',')
    for model_name in models:
        print("Predicting with model:", model_name)
        set_arm(args, model_name)
        create_id(args)

        predict_and_save_anndata_scgen(args, adata, model_name)

    return