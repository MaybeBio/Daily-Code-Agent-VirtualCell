"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


class ReconstructionTaskMixin:
    def adapter_finetuning_reconstruction(self, cfgs):
        """提取原始数据和嵌入 + 训练验证测试集划分 + 微调 + 重建"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            raw_exprs = raw_exprs[mask]
            labels = labels[mask]
            cell_ids = cell_ids[mask]

        X_train, X_val, X_test, split = self.evaluation_split(cell_ids, 'reconstruction')

        scaler = StandardScaler()
        cell_embs_train_scaled = scaler.fit_transform(cell_embs[X_train])
        cell_embs_val_scaled = scaler.transform(cell_embs[X_val])
        cell_embs_test_scaled = scaler.transform(cell_embs[X_test])

        cell_embs_scaled = np.zeros_like(cell_embs)
        cell_embs_scaled[X_train] = cell_embs_train_scaled
        cell_embs_scaled[X_val] = cell_embs_val_scaled
        cell_embs_scaled[X_test] = cell_embs_test_scaled

        # Preserve the full output head; only the selected training HVGs are
        # supervised, exactly as in the original full-width boolean masks.
        from bio.data.preprocess import hvg_indices
        selected_gene_indices = hvg_indices(
            self.adata_all[np.asarray(cell_ids)[X_train]], n_top=400)
        gene_index = torch.as_tensor(selected_gene_indices, dtype=torch.long, device=self.device)
        cell_embs_tensor = torch.from_numpy(np.asarray(cell_embs_scaled, dtype=np.float32))
        targets = torch.from_numpy(np.asarray(raw_exprs[:, selected_gene_indices], dtype=np.float32))

        def loader(rows, shuffle=False):
            return DataLoader(TensorDataset(cell_embs_tensor[rows], targets[rows]),
                              batch_size=64, shuffle=shuffle)

        self.train_loader = loader(X_train, shuffle=True)
        self.valid_loader = loader(X_val)
        self.test_loader = loader(X_test)

        best_metric = float("inf")
        no_improve = 0
        patience = 5
        delta = 1e-4
        best_state_dict = None

        for epoch in range(int(cfgs.get("model_cfg", {}).get("head_epochs", 500))):
            self.restruct.train()
            running_loss = 0.0
            batch_count = 0
            for cell_emb, expr in self.train_loader:
                cell_emb = cell_emb.to(self.device)
                expr = expr.to(self.device)

                pred = self.forward_reconstruction(cell_emb)
                loss = F.mse_loss(pred.index_select(1, gene_index), expr)

                self.optimizer_restruct.zero_grad()
                loss.backward()
                self.optimizer_restruct.step()

                running_loss += loss.item()
                batch_count += 1

            if (epoch + 1) % 20 == 0:
                self.restruct.eval()
                recon_list, truth_list = [], []

                with torch.no_grad():
                    for cell_emb, expr in self.valid_loader:
                        cell_emb = cell_emb.to(self.device)
                        expr = expr.to(self.device)

                        pred = self.forward_reconstruction(cell_emb)

                        pred = pred.index_select(1, gene_index)
                        recon_list.append(pred.cpu().numpy())
                        truth_list.append(expr.cpu().numpy())

                valid_input = {
                    "recon": np.concatenate(recon_list, 0),
                    "truth": np.concatenate(truth_list, 0),
                }

                val_metric = mean_squared_error(valid_input["truth"], valid_input["recon"])

                if val_metric < best_metric - delta:
                    best_metric = val_metric
                    best_state_dict = copy.deepcopy(self.restruct.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

        if best_state_dict is not None:
            self.restruct.load_state_dict(best_state_dict)

        self.restruct.eval()
        recon_list, truth_list = [], []

        with torch.no_grad():
            for cell_emb, expr in self.test_loader:
                cell_emb = cell_emb.to(self.device)
                expr = expr.to(self.device)

                pred = self.forward_reconstruction(cell_emb)

                pred = pred.index_select(1, gene_index)
                recon_list.append(pred.cpu().numpy())
                truth_list.append(expr.cpu().numpy())

        recon = np.concatenate(recon_list, 0)
        truth = np.concatenate(truth_list, 0)
        test_input = {
            "recon": recon,
            "truth": truth,
            "split": split,
        }
        final_score = self.metric(test_input, task="reconstruction")
        return recon, truth
