"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class AnnotationTaskMixin:
    def adapter_finetuning_annotation(self, cfgs):
        """提取嵌入 + 训练测试集划分 + 数据标准化 + 微调 + 注释"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            labels = labels[mask]
            cell_ids = cell_ids[mask]

        train, val, test, split = self.evaluation_split(cell_ids, 'annotation')
        X_train, X_val, X_test = cell_embs[train], cell_embs[val], cell_embs[test]
        y_train, y_val, y_test = labels[train], labels[val], labels[test]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
        self.valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        epochs = int(cfgs.get("model_cfg", {}).get("head_epochs", 500))
        if epochs < 1:
            raise ValueError('Annotation head_epochs must be positive')
        best_loss = float('inf')
        best_state = None
        best_epoch = None
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.classify.train()
            for inputs, label in self.train_loader:
                if inputs.size(0) == 1:
                    continue
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                logit = self.forward_annotation(inputs)
                loss = self.loss(logit, label)

                self.optimizer_classify.zero_grad()
                loss.backward()
                self.optimizer_classify.step()

                running_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            # Select on held-out validation cross entropy; evaluate test only once.
            self.classify.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for inputs, label in self.valid_loader:
                    inputs, label = inputs.to(self.device), label.to(self.device)
                    loss = self.loss(self.forward_annotation(inputs), label)
                    val_loss += loss.item() * label.size(0)
                    n_val += label.size(0)
            if n_val == 0:
                raise ValueError('Annotation validation split is empty')
            val_loss /= n_val
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = copy.deepcopy(self.classify.state_dict())
        if best_state is None:
            raise ValueError('Annotation produced no finite validation checkpoint')
        self.classify.load_state_dict(best_state)
        split.update(annotation_selection='validation_cross_entropy',
                     best_epoch=best_epoch, best_validation_loss=best_loss)

        test_preds = []
        test_probabilities = []
        test_labels = []
        self.classify.eval()
        with torch.no_grad():
            for inputs, label in self.test_loader:
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                logit = self.forward_annotation(inputs)
                pred = torch.softmax(logit, dim=1).argmax(dim=1)
                test_probabilities.append(torch.softmax(logit, dim=1).cpu().numpy())
                test_preds.append(pred.cpu().numpy())
                test_labels.append(label.cpu().numpy())

        preds = np.concatenate([x.ravel() for x in test_preds])
        labels = np.concatenate([x.ravel() for x in test_labels])
        input_dict = {"pred":preds, "label":labels, "probabilities": np.concatenate(test_probabilities),
                      "classes": np.arange(self.cell_type_num), "split": split}

        metric = self.metric(input_dict, "annotation")
        return labels, preds

    def full_finetuning_annotation(self, cfgs):
        print("start full finetuning annotation")

        for epoch in range(500):
            embeddings = []
            logits = []
            labels = []
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs in self.train_loader:
                logit, label, cell_emb = self.forward(inputs)
                loss = self.loss(logit,label)
                loss.backward()

                if hasattr(self, 'classify') and self.classify is not None:
                    self.optimizer_classify.step()
                    self.optimizer_classify.zero_grad()

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        preds = []
        labels = []
        cell_embs = []
        with torch.no_grad():
            if self.model_name not in ["PCA_model","UMAP","scVI"]:
                self.model.eval()
            for inputs in self.test_loader:
                logit, label, cell_emb = self.forward(inputs)
                pred = torch.softmax(logit, dim=1).argmax(dim=1)
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
                cell_embs.append(cell_emb.cpu().numpy())

        preds = np.concatenate([x.ravel() for x in preds])
        labels = np.concatenate([x.ravel() for x in labels])
        cell_emb = np.concatenate(cell_embs,0)
        input_dict = {"pred":preds, "label":labels, "cell_emb":cell_emb}
        metric = self.metric(input_dict, "annotation")

        return np.vstack(cell_embs), np.hstack(labels), np.hstack(preds)
