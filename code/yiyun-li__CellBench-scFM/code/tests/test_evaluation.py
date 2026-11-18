import unittest
from unittest.mock import patch
import numpy as np
from sklearn import metrics
from bio.metrics import annotation, reconstruction, perturbation, correction, clustering


class EvaluationTests(unittest.TestCase):
    def test_precision_recall_order_and_probabilities(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0, 1, 1, 1, 1])
        prob = np.array([[.9,.1],[.4,.6],[.3,.7],[.1,.9],[.2,.8]])
        r = annotation(p, y, prob, [0, 1])
        self.assertAlmostEqual(r['precision_macro'], metrics.precision_score(y,p,average='macro'))
        self.assertAlmostEqual(r['recall_macro'], metrics.recall_score(y,p,average='macro'))
        self.assertNotEqual(r['precision_macro'], r['recall_macro'])
        self.assertAlmostEqual(r['AUROC_macro'],1.)
        self.assertAlmostEqual(r['AUPRC_macro'],1.)

    def test_absent_classes_and_invalid_probabilities(self):
        r = annotation([0,0], [0,0], [[1,0],[1,0]], [0,1])
        self.assertIsNone(r['AUROC_macro'])
        self.assertEqual(r['absent_test_classes'], ['1'])
        with self.assertRaises(ValueError): annotation([0],[0],[[1,1]],[0,1])

    def test_constant_correlations_are_missing_not_perfect(self):
        r = reconstruction(np.ones((3,4)), np.ones((3,4)), 50)
        self.assertIsNone(r['Pearson_expr'])
        self.assertEqual(r['Pearson_expr_valid_n'],0)
        self.assertEqual(r['top_k'],4)
        self.assertEqual(r['MSE_expr'],0)
        self.assertNotIn('Sign_agree',r)
        self.assertNotIn('AvgGR',r)

    def test_correlations_have_explicit_axes(self):
        x = np.array([[1,2,4],[2,4,8],[3,6,12]])
        r=reconstruction(x,x)
        self.assertAlmostEqual(r['Pearson_expr'],1)
        self.assertAlmostEqual(r['Pearson_gene'],1)
        with self.assertRaises(ValueError): reconstruction([[np.nan]],[[1]])

    def test_condition_means_equal_weight_not_cell_weight(self):
        t=np.array([[1,2],[1,2],[1,2],[3,4]])
        p=np.array([[2,3],[2,3],[2,3],[6,7]])
        r=perturbation(p,t,np.zeros_like(t),['a','a','a','b'])
        self.assertAlmostEqual(r['MAE_expr'],2.)
        self.assertAlmostEqual(r['MAE_delta'],2.)
        self.assertEqual(r['n_groups'],2)
        self.assertNotIn('MSE_LFC',r)
        self.assertNotIn('AvgPP',r)

    def test_no_effect_not_counted_as_direction_success(self):
        x=np.ones((3,4))
        r=perturbation(x,x,x,['a','a','b'])
        self.assertIsNone(r['Sign_agree_changed'])
        self.assertIsNone(r['TopK_directional_recall'])
        self.assertIsNone(r['DES'])
        self.assertIsNone(r['MSE_skill_vs_control'])

    def test_direction_recall_requires_actual_downregulation(self):
        r=perturbation([[3,3,3]],[[1,3,2]],[[2,2,2]],['a'],top_k=50)
        self.assertAlmostEqual(r['Sign_agree_changed'],.5)
        self.assertAlmostEqual(r['TopK_directional_recall'],.5)

    def test_des_is_vcc_fdr_overlap_at_n(self):
        n_c,n_p,g=30,20,6
        ctrl=np.zeros((n_c,g)); truth=np.zeros((n_p,g)); pred=np.zeros((n_p,g))
        truth[:,0]=5; truth[:,1]=5
        pred[:,0]=5; pred[:,2]=5
        groups=np.array(['["knock","T"]']*n_p)
        r=perturbation(pred,truth,np.zeros((n_p,g)),groups,
                       control_cells=ctrl, control_labels=np.array(['T']*n_c))
        self.assertEqual(r['DES_definition'],'vcc_overlap_at_N_fdr_bh')
        self.assertAlmostEqual(r['DES'],.5)
        self.assertAlmostEqual(r['MAE_delta'], np.mean(np.abs((pred.mean(0)-ctrl.mean(0))-(truth.mean(0)-ctrl.mean(0)))))

    def test_perfect_de_genes_give_des_one(self):
        n_c,n_p,g=30,20,4
        ctrl=np.zeros((n_c,g)); x=np.zeros((n_p,g)); x[:,0]=4; x[:,1]=4
        groups=np.array(['["knock","T"]']*n_p)
        r=perturbation(x,x,np.zeros((n_p,g)),groups,
                       control_cells=ctrl, control_labels=np.array(['T']*n_c))
        self.assertAlmostEqual(r['DES'],1.)

    def test_fixed_clustering_does_not_optimize_reference_labels(self):
        rng=np.random.default_rng(41)
        x=np.r_[rng.normal(-4,.1,(20,3)),rng.normal(4,.1,(20,3))]
        with patch('scib.metrics.clustering.opt_louvain',side_effect=AssertionError('oracle used')):
            r,_=clustering(x,np.repeat([0,1],20))
        self.assertEqual(r['clustering_mode'],'fixed')
        self.assertGreater(r['ari'],.9)

    def test_batch_asw_uses_cell_type_strata(self):
        # Each cell type has overlapping batches; types are far apart.
        rng=np.random.default_rng(2)
        x=np.r_[rng.normal(-4,.1,(20,3)),rng.normal(4,.1,(20,3))]
        y=np.repeat(['A','B'],20); b=np.tile(['a','b'],20)
        with patch('scib.metrics.ilisi_graph',return_value=.8), patch('scib.metrics.clisi_graph',return_value=.9):
            r=correction(x,b,y,{'compute_kbet':False})
        self.assertGreater(r['Batch_ASW'],.8)
        self.assertEqual(r['iLISI_scaled'],.8)
        self.assertEqual(r['cLISI_scaled'],.9)
        self.assertIsNone(r['kBET'])
        self.assertNotIn('AvgBC',r)

    def test_kbet_is_finite_without_rpy2_and_higher_when_batches_mix(self):
        rng=np.random.default_rng(7)
        mixed=rng.normal(size=(80,4)); mixed_y=np.repeat(['A','B'],40); mixed_b=np.tile(['x','y'],40)
        mixed_r=correction(mixed,mixed_b,mixed_y)
        self.assertIsInstance(mixed_r['kBET'],float)
        self.assertGreaterEqual(mixed_r['kBET'],0)
        self.assertLessEqual(mixed_r['kBET'],1)
        self.assertEqual(mixed_r['metric_status']['kBET'],'ok')
        split=np.r_[rng.normal(-5,.05,(40,4)),rng.normal(5,.05,(40,4))]
        split_y=np.tile(['A','B'],40); split_b=np.repeat(['x','y'],40)
        split_r=correction(split,split_b,split_y)
        self.assertIsInstance(split_r['kBET'],float)
        self.assertGreater(mixed_r['kBET'],split_r['kBET'])

    def test_optional_failure_is_not_a_numeric_score(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(40,3)); y=np.repeat(['A','B'],20); b=np.tile(['a','b'],20)
        with patch('scib.metrics.ilisi_graph',side_effect=RuntimeError('missing binary')), patch('scib.metrics.clisi_graph',return_value=.9):
            r=correction(x,b,y,{'compute_kbet':False})
        self.assertIsNone(r['iLISI_scaled'])
        self.assertIn('missing binary',r['metric_status']['iLISI_scaled'])

    def test_perturbation_head_can_predict_downregulation(self):
        import torch
        from bio.models.heads import PerturbDecoder
        from bio.models.adapters.cellplm import CellPLM
        from types import SimpleNamespace
        head=PerturbDecoder(3,2,4).eval()
        with torch.no_grad():
            head.proj_out.weight.zero_(); head.proj_out.bias.fill_(-.5)
            out=CellPLM.forward_perturbation(SimpleNamespace(pert_decoder=head),torch.zeros(2,3),torch.zeros(2,dtype=torch.long),torch.ones(2,4))
        torch.testing.assert_close(out,torch.full((2,4),.5))



class ExportAndVersionTests(unittest.TestCase):
    def test_five_tasks_have_four_figure_metrics(self):
        from bio.io.export import PRIMARY
        self.assertEqual(list(PRIMARY), ['clustering', 'annotation', 'reconstruction', 'perturbation', 'correction'])
        for task in PRIMARY:
            self.assertEqual(len(PRIMARY[task]), 4, msg=task)
        self.assertEqual(list(PRIMARY['perturbation']), ['Pearson_delta', 'Spearman_delta', 'MAE_delta', 'DES'])

    def test_disease_panel_ranks_four_datasets_equally(self):
        import pandas as pd
        from bio.io.export import paired_ranks
        records = []
        for dataset, score_a, score_b in (
            ('ms', 0.9, 0.1),
            ('covid_lung', 0.9, 0.1),
            ('tnbc', 0.1, 0.9),
            ('uc_epi', 0.1, 0.9),
        ):
            records.append(dict(dataset=dataset, model='a', seed=41, topk=0, phase='adapter', protocol='v2', Pearson_delta=score_a))
            records.append(dict(dataset=dataset, model='b', seed=41, topk=0, phase='adapter', protocol='v2', Pearson_delta=score_b))
        rank, coverage = paired_ranks(pd.DataFrame(records), {'Pearson_delta': True})
        self.assertEqual(int(coverage.complete.sum()), 4)
        self.assertAlmostEqual(float(rank.loc[rank.model == 'a', 'mean_metric_rank'].iloc[0]), 1.5)
        self.assertAlmostEqual(float(rank.loc[rank.model == 'b', 'mean_metric_rank'].iloc[0]), 1.5)

    def test_export_without_panel_writes_each_section_separately(self):
        import json, tempfile
        from pathlib import Path
        from bio.io.export import export_panels
        from bio.metrics import METRIC_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, model, value in (
                ('bmmc', 'a', 0.9), ('bmmc', 'b', 0.1),
                ('ms', 'a', 0.2), ('ms', 'b', 0.8),
            ):
                folder = root / 'metrics' / 'clustering'
                folder.mkdir(parents=True, exist_ok=True)
                record = dict(metric_version=METRIC_VERSION, dataset=dataset, model=model, seed=41,
                              topk=0, phase='adapter', updated_at='2026-01-01', run_id='x',
                              config={'evaluation_cfg': {}, 'data_cfg': {}},
                              summary={'ari': value, 'nmi': value, 'asw': value, 'acc': value})
                (folder / f'{dataset}_{model}.json').write_text(json.dumps(record))
            written = export_panels('clustering', tmp)
            self.assertIn('organ', written)
            self.assertIn('disease', written)
            organ = (root / 'exports' / 'v2' / 'organ' / 'clustering_scores.csv').read_text()
            self.assertIn('bmmc', organ)
            self.assertNotIn('ms', organ)

    def test_ranking_is_scale_invariant_and_excludes_incomplete_blocks(self):
        import pandas as pd
        from bio.io.export import paired_ranks
        records=[]
        for dataset in ['d1','d2']:
            for model, corr, mse in [('a',.8,1.),('b',.7,2.)]:
                records.append(dict(dataset=dataset,model=model,seed=41,topk=100,phase='adapter',protocol='v2',Pearson_expr=corr,MSE_expr=mse))
        frame=pd.DataFrame(records)
        primary={'Pearson_expr':True,'MSE_expr':False}
        original,_=paired_ranks(frame,primary)
        frame['MSE_expr']*=10000
        rescaled,_=paired_ranks(frame,primary)
        pd.testing.assert_frame_equal(original,rescaled)
        frame.loc[frame.dataset=='d2','MSE_expr']=np.nan
        rank, coverage=paired_ranks(frame,primary)
        self.assertEqual(int(coverage.complete.sum()),1)
        self.assertEqual(rank.iloc[0]['model'],'a')

    def test_legacy_records_excluded_by_default(self):
        import json, tempfile
        from pathlib import Path
        from bio.io.results import log_metrics, load_runs
        with tempfile.TemporaryDirectory() as root:
            cfg={'output_dir':root,'data_cfg':{'dataset_name':'d'},'model_cfg':{'model':'m','tasks':'annotation'}}
            path=log_metrics(cfg,{'acc':1})
            legacy=json.loads(path.read_text()); legacy.pop('metric_version')
            (Path(path).parent/'old.json').write_text(json.dumps(legacy))
            self.assertEqual(len(load_runs('annotation',root)),1)
            self.assertEqual(len(load_runs('annotation',root,metric_version=1)),1)


class PerturbationArtifactTests(unittest.TestCase):
    def test_evaluation_keeps_all_same_type_controls_and_prediction_order(self):
        from bio.data.perturbation import create_evaluation_adata
        raw=np.arange(12,dtype=float).reshape(6,2)
        labels=np.array([0,0,0,0,1,1])
        conditions=np.array(['ctrl','ctrl','p1','p2','ctrl','p1'])
        controls=np.array([1,1,0,0,1,0])
        pred,truth=create_evaluation_adata([3,2],{3:0,2:0},raw,np.array([[30,31],[20,21]]),labels,np.arange(6),conditions,controls)
        self.assertEqual(truth.obs.cell_id.tolist(),[0,1,2,3])
        np.testing.assert_array_equal(pred.X[2:4],[[20,21],[30,31]])
        np.testing.assert_array_equal(pred.X[:2],raw[:2])


if __name__ == '__main__': unittest.main()
