

Nicheformer integrates Weights & Biases (W&B) for comprehensive experiment tracking across pretraining and fine-tuning workflows. This integration enables systematic monitoring of model performance, hyperparameter comparison, and reproducibility management for both self-supervised learning and downstream tasks.

## W&B Logger Architecture

The experiment tracking architecture leverages PyTorch Lightning's native W&B integration, creating separate logging contexts for pretraining and fine-tuning phases. This separation allows organized experiment comparison across different biological contexts and training objectives.

```mermaid
flowchart TD
    A[Training Script] --> B[Config File]
    B --> C{Training Phase}
    C -->|Pretraining| D[WandbLogger<br/>Project: FM-{organ}]
    C -->|Fine-tuning| E[WandbLogger<br/>Project: Get_predictions_{organ}<br/>Run Name: dynamic]
    D --> F[Trainer Integration]
    E --> F
    F --> G[Model training_step/validation_step]
    G --> H[Metrics Logged to W&B]
    H --> I[Callbacks: Checkpoint & LR Monitor]
```

The logger initialization follows a consistent pattern across both training workflows. In the pretraining pipeline, the logger is configured with a project name that incorporates the organ-specific dataset identifier (`FM-{organ}`), while the fine-tuning pipeline uses a more descriptive project convention (`Get_predictions_{organ}`) with dynamically generated run names that capture experiment specifics.

Sources: [_train.py](src/nicheformer/_train.py#L4-L38), [_fine_tune.py](src/nicheformer/_fine_tune.py#L5-L49)

## Pretraining Experiment Tracking

For pretraining experiments, W&B captures the complete masked language modeling training process. The logger configuration automatically tracks model parameters, system metrics, and custom loss metrics without requiring explicit W&B API calls.

### Logger Configuration

The pretraining W&B logger is initialized in `manual_train_fm()` with project-scoped naming and entity organization:
- **Project**: `FM-{organ}` where `{organ}` is the dataset configuration parameter
- **Entity**: `nicheformer` (centralized team account)
- **Automatic Parameter Tracking**: All model hyperparameters are logged via PyTorch Lightning's automatic parameter capture

Sources: [_train.py](src/nicheformer/_train.py#L38)

### Training Metrics

The Nicheformer model logs metrics at both training and validation steps using PyTorch Lightning's `self.log()` method. The masked language modeling loss serves as the primary training signal:
- **train_loss**: Averaged across all GPUs during training steps
- **val_loss**: Computed on validation data during validation epochs

The logging configuration ensures proper handling in distributed training environments through `sync_dist=True`, which aggregates metrics from all GPU workers before logging to W&B.

Sources: [models/_nicheformer.py](src/nicheformer/models/_nicheformer.py#L172), [models/_nicheformer.py](src/nicheformer/models/_nicheformer.py#L191)

### Learning Rate Monitoring

A dedicated learning rate callback tracks optimization dynamics throughout training:
- **LearningRateMonitor**: Captures learning rate changes at step-level intervals
- **Logging Interval**: Per-step logging provides granular visibility into warmup and annealing schedules
- **W&B Integration**: Learning rate values appear as time-series metrics in the W&B dashboard

Sources: [_train.py](src/nicheformer/_train.py#L41)

> [!TIP]
> The combination of per-step learning rate tracking with per-step loss logging enables direct correlation between learning rate schedule changes and training convergence patterns in W&B visualizations.

## Fine-tuning Experiment Tracking

Fine-tuning experiments extend the W&B integration with task-specific metrics and dynamic run naming for systematic experiment organization across multiple downstream tasks and biological contexts.

### Dynamic Run Naming

The fine-tuning workflow generates descriptive run names that encode critical experiment hyperparameters, enabling easy identification and comparison:
- **Supervised Task Type**: `supervised_task` parameter (e.g., niche_classification, density_regression)
- **Checkpoint Steps**: Extracted from the pretrained checkpoint path
- **Baseline Flag**: Appended `_baseline` when training without the pretrained backbone

This naming convention allows grouping experiments by task while preserving unique identifiers for each configuration variant.

Sources: [_fine_tune.py](src/nicheformer/_fine_tune.py#L42-L49)

### Task-Specific Metrics

Fine-tuning models log distinct metrics based on the supervised learning objective:

| Task Type | Training Metrics | Validation Metrics |
|-----------|-----------------|-------------------|
| Regression | `fine_tuning_regression_train` | `fine_tuning_regression_validation` |
| Classification | `fine_tuning_classification_train`, `accuracy_train` | `fine_tuning_classification_validation`, `accuracy_validation` |

Regression tasks predict continuous values such as gene expression or niche density, while classification tasks target discrete categories like cell types or tissue regions. The metrics pipeline automatically adapts logging behavior to the task type specified in the configuration.

Sources: [models/_fine_tune_model.py](src/nicheformer/models/_fine_tune_model.py#L189-L212), [models/_fine_tune_model.py](src/nicheformer/models/_fine_tune_model.py#L238-L257)

### Specialized Handling Options

The fine-tuning metrics pipeline accommodates specialized modeling scenarios through conditional logging logic:

- **Binary Classification**: Labels are binarized (non-zero values become 1) for niche_binary_classification tasks
- **Zero-Label Masking**: Zero labels are mapped to -100 for ignored labels in classification tasks with `ignore_zeros=True`
- **Density Prediction**: Labels are summed across dimensions when predicting cell density with `predict_density=True`
- **Distribution Regression**: Labels are softmax-transformed when modeling probability distributions

These conditional paths ensure that metrics reflect the appropriate loss computation and evaluation logic for each experimental setup.

Sources: [models/_fine_tune_model.py](src/nicheformer/models/_fine_tune_model.py#L198-L212)

## Trainer Configuration for W&B

Both pretraining and fine-tuning workflows configure the PyTorch Lightning Trainer with W&B-specific parameters that control logging frequency and granularity:

### Logging Frequency

- **log_every_n_steps=100**: Metrics are logged to W&B every 100 training steps, balancing storage overhead with temporal resolution
- **check_val_every_n_epoch=50**: Pretraining validation metrics are captured every 50 epochs, with fine-tuning disabling explicit validation checkpoints for prediction-focused workflows

These settings define the temporal resolution of W&B tracking without requiring additional code in model training loops.

Sources: [_train.py](src/nicheformer/_train.py#L48-L49), [_fine_tune.py](src/nicheformer/_fine_tune.py#L59)

### Distributed Training Considerations

The W&B integration properly handles multi-GPU training through PyTorch Lightning's distributed coordination:
- **sync_dist=True**: Ensures metrics are averaged across all GPUs before logging
- **reduce_fx='mean'**: Uses mean aggregation for distributed metrics
- **prog_bar=True**: Optionally displays metrics in the training progress bar

This configuration prevents duplicate metric logging from individual GPU workers and ensures W&B receives globally aggregated values.

Sources: [models/_nicheformer.py](src/nicheformer/models/_nicheformer.py#L172), [models/_fine_tune_model.py](src/nicheformer/models/_fine_tune_model.py#L189)

## Configuration Management and W&B

Experiment configuration is centralized in Python dictionary files that W&B automatically logs as hyperparameters. This approach enables systematic hyperparameter comparison across experiments through W&B's parameter sweep and comparison tools.

### Pretraining Configuration

The pretraining configuration in `_config_train.py` includes model architecture, optimization, and data parameters that W&B captures automatically:
- Architecture: `dim_model`, `nheads`, `nlayers`, `dim_feedforward`
- Training: `lr`, `warmup`, `batch_size`, `max_epochs`
- Tokenization: `masking_p`, `n_tokens`, `context_length`
- Dataset: `organ`, `specie`, `assay`, `modality`

These parameters define the entire experimental setup and become searchable dimensions in W&B.

Sources: [config_files/_config_train.py](src/nicheformer/config_files/_config_train.py#L1-L25)

### Fine-tuning Configuration

Fine-tuning configurations in `_config_fine_tune.py` extend the parameter set with task-specific options:
- Task Definition: `supervised_task`, `n_classes`, `dim_prediction`
- Transfer Learning: `checkpoint_path`, `freeze`, `reinit_layers`, `extract_layers`
- Specialized Features: `predict_density`, `regress_distribution`, `ignore_zeros`
- Baseline Configuration: `baseline` flag for ablation studies

The parameter naming convention enables consistent organization across different fine-tuning objectives and biological datasets.

Sources: [config_files/_config_fine_tune.py](src/nicheformer/config_files/_config_fine_tune.py#L1-L20)

## Experiment Organization Strategy

The W&B project structure enables organized experiment comparison across multiple dimensions:

### Project Organization

| Project Type | Naming Convention | Purpose |
|-------------|-------------------|---------|
| Pretraining | `FM-{organ}` | Self-supervised masked language modeling experiments |
| Fine-tuning | `Get_predictions_{organ}` | Supervised downstream task experiments |

The organ-based scoping allows filtering experiments by biological context (e.g., brain, liver, lung) while maintaining separation between pretraining and fine-tuning phases.

Sources: [_train.py](src/nicheformer/_train.py#L38), [_fine_tune.py](src/nicheformer/_fine_tune.py#L49)

### Run Naming Convention

Fine-tuning run names encode the complete experimental configuration:
```
{supervised_task}_{checkpoint_steps}_steps[{_baseline}]
```

Examples:
- `niche_classification_1500000_steps` - Standard fine-tuning from 1.5M-step checkpoint
- `density_regression_2500000_steps_baseline` - Baseline ablation from 2.5M-step checkpoint

This naming strategy enables rapid experiment identification and supports W&B's run grouping and comparison features.

Sources: [_fine_tune.py](src/nicheformer/_fine_tune.py#L42-L48)

## Practical Implementation

Implementing W&B tracking in custom Nicheformer workflows requires three key components: logger initialization, metric logging, and trainer configuration.

### Basic Integration Pattern

```python
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Initialize logger
wandb_logger = WandbLogger(
    project=f'FM-{config["organ"]}',
    entity='nicheformer'
)

# Configure callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='/output',
    every_n_train_steps=1500,
    monitor='train_loss',
    save_top_k=-1
)
lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize trainer with W&B
trainer = pl.Trainer(
    logger=wandb_logger,
    log_every_n_steps=100,
    callbacks=[checkpoint_callback, lr_monitor],
    # ... other trainer parameters
)
```

Sources: [_train.py](src/nicheformer/_train.py#L38-L55)

### Custom Metric Logging

Custom metrics are logged within PyTorch Lightning's `training_step` and `validation_step` methods:

```python
def training_step(self, batch, batch_idx):
    # Compute predictions and loss
    loss = self._compute_loss(predictions, targets)
    
    # Log to W&B with distributed synchronization
    self.log('custom_metric_name', 
             loss, 
             sync_dist=True, 
             prog_bar=True,
             reduce_fx='mean')
    
    return loss
```

The `sync_dist=True` parameter ensures proper metric aggregation in distributed training environments.

Sources: [models/_nicheformer.py](src/nicheformer/models/_nicheformer.py#L156-L173)

> [!TIP]
> When logging custom metrics in distributed training, always use `reduce_fx='mean'` with `sync_dist=True` to ensure W&B receives globally averaged values rather than per-GPU duplicates.

## Next Steps

For comprehensive experiment management workflows, continue with:
- [Configuration Files and Parameters](26-configuration-files-and-parameters) - Deep dive into parameter organization and hyperparameter management
- [Managing Multi-modal and Multi-organ Experiments](28-managing-multi-modal-and-multi-organ-experiments) - Strategies for scaling W&B tracking across diverse biological datasets

For implementation guidance:
- [Pretraining Nicheformer on Large-Scale Data](15-pretraining-nicheformer-on-large-scale-data) - Complete pretraining workflow with W&B integration
- [Fine-Tuning Model Architecture](18-fine-tuning-model-architecture) - Fine-tuning experiment setup with task-specific tracking