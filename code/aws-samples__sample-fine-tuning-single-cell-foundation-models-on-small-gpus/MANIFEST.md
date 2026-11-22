# 代码快照清单

- 上游仓库: https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus
- 快照日期: 2026-11-22
- 快照 commit: [`e1765d3d23bb`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/commit/e1765d3d23bbd02e3552402e2defb0621f65bdce)
- 归档策略: `repo_copy=code` — **代码摘录,不是完整仓库**。保留源码、构建/配置文件、README、文档;数据、二进制、图片、权重不保留。

## 保留 (104 个文件, 1.5 MB)

| 类别 | 文件数 | 大小 |
|---|---:|---:|
| 构建/配置 | 20 | 826.7 KB |
| 源码 | 72 | 485.8 KB |
| 文档 | 10 | 111.4 KB |
| README | 2 | 67.6 KB |

按后缀: `.py` 67 (456.7 KB)、`.txt` 7 (445.5 KB)、`.lock` 1 (335.4 KB)、`.md` 11 (178.1 KB)、`.yaml` 9 (30.6 KB)、`.sh` 5 (29.0 KB)、`Dockerfile` 1 (6.8 KB)、`.yml` 1 (5.0 KB)、`pyproject.toml` 1 (3.5 KB)、`无后缀` 1 (947 B)

> README / 文档中的相对图片与链接已改写为 GitHub 绝对地址。

## 省略 (20 个文件, 25.4 KB)

| 原因 | 文件数 | 大小 |
|---|---:|---:|
| 后缀不在白名单 | 20 | 25.4 KB |

体积最大的省略文件:

| 路径 | 大小 |
|---|---:|
| [`sagemaker/Dockerfile.scgpt`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/Dockerfile.scgpt) | 8.4 KB |
| [`docs/iam/sagemaker-launcher-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-launcher-policy.json) | 2.4 KB |
| [`docs/iam/sagemaker-execution-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-execution-policy.json) | 2.3 KB |
| [`docs/security/log-bucket-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/security/log-bucket-policy.json) | 2.1 KB |
| [`docs/iam/codebuild-service-role-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/codebuild-service-role-policy.json) | 1.6 KB |
| [`docs/security/s3-bucket-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/security/s3-bucket-policy.json) | 1.5 KB |
| [`sagemaker/requirements-scgpt.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/requirements-scgpt.in) | 1.4 KB |
| [`scripts/requirements-launcher.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/scripts/requirements-launcher.in) | 807 B |
| [`sagemaker/requirements-toolkit.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/requirements-toolkit.in) | 708 B |
| [`sagemaker/requirements-scgpt-build.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/requirements-scgpt-build.in) | 578 B |
| [`docs/iam/sagemaker-execution-vpc-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-execution-vpc-policy.json) | 544 B |
| [`sagemaker/requirements-toolkit-build.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/requirements-toolkit-build.in) | 532 B |
| [`.gitignore`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/.gitignore) | 474 B |
| [`.dockerignore`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/.dockerignore) | 420 B |
| [`docs/iam/sagemaker-execution-weights-kms-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-execution-weights-kms-policy.json) | 389 B |
| [`docs/iam/codebuild-trust-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/codebuild-trust-policy.json) | 376 B |
| [`docs/iam/sagemaker-trust-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-trust-policy.json) | 363 B |
| [`docs/iam/sagemaker-execution-kms-policy.json`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/docs/iam/sagemaker-execution-kms-policy.json) | 334 B |
| [`sagemaker/requirements-bootstrap.in`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/requirements-bootstrap.in) | 298 B |
| [`sagemaker/scgpt_weights/.gitkeep`](https://github.com/aws-samples/sample-fine-tuning-single-cell-foundation-models-on-small-gpus/raw/HEAD/sagemaker/scgpt_weights/.gitkeep) | 0 B |

> 省略规则:普通源码 >2 MB、文档/构建配置 >1 MB(README 与 .ipynb 不设限),或后缀不在白名单(图片/权重/数据表/二进制/通用 .txt/.log/.csv/.json/.xml 等)、以及子模块。需要全文时按上表链接直接取回,或用 git 克隆上游仓库。

> 本清单由 `scripts/archive.py` 每次归档时自动生成并覆盖。
