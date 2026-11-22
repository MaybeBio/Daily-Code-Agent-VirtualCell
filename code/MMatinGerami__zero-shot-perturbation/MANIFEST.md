# 代码快照清单

- 上游仓库: https://github.com/MMatinGerami/zero-shot-perturbation
- 快照日期: 2026-11-22
- 快照 commit: [`2276adcd121f`](https://github.com/MMatinGerami/zero-shot-perturbation/commit/2276adcd121f9bf87dd9fc857c879f2a8b3d4e71)
- 归档策略: `repo_copy=code` — **代码摘录,不是完整仓库**。保留源码、构建/配置文件、README、文档;数据、二进制、图片、权重不保留。

## 保留 (111 个文件, 792.8 KB)

| 类别 | 文件数 | 大小 |
|---|---:|---:|
| 构建/配置 | 64 | 473.2 KB |
| 源码 | 38 | 240.8 KB |
| 文档 | 8 | 73.8 KB |
| README | 1 | 4.9 KB |

按后缀: `.lock` 1 (410.9 KB)、`.py` 37 (238.7 KB)、`.md` 8 (77.7 KB)、`.yaml` 59 (58.9 KB)、`.sh` 1 (2.0 KB)、`Makefile` 1 (1.4 KB)、`无后缀` 1 (1.0 KB)、`pyproject.toml` 1 (1017 B)、`Dockerfile` 1 (674 B)、`.yml` 1 (273 B)

> README / 文档中的相对图片与链接已改写为 GitHub 绝对地址。

## 省略 (394 个文件, 11.0 MB)

| 原因 | 文件数 | 大小 |
|---|---:|---:|
| 后缀不在白名单 | 394 | 11.0 MB |

体积最大的省略文件:

| 路径 | 大小 |
|---|---:|
| [`results/tables/fluctuation_blend.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/tables/fluctuation_blend.csv) | 1.8 MB |
| [`results/tables/coverage_genes.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/tables/coverage_genes.csv) | 1.8 MB |
| [`results/tables/uncertainty_selective.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/tables/uncertainty_selective.csv) | 1.1 MB |
| [`results/tables/transferability_perturbations.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/tables/transferability_perturbations.csv) | 538.1 KB |
| [`results/figures/uncertainty.png`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/figures/uncertainty.png) | 315.7 KB |
| [`results/local_eval/h1_norm_restored/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_norm_restored/results.csv) | 80.9 KB |
| [`results/local_eval/h1_norm_restored+cd4/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_norm_restored%2Bcd4/results.csv) | 80.9 KB |
| [`results/local_eval/h1_mean_transfer/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_mean_transfer/results.csv) | 80.8 KB |
| [`results/local_eval/h1_v5eq_iid/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v5eq_iid/results.csv) | 80.8 KB |
| [`results/local_eval/h1_v5eq_iid_de/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v5eq_iid_de/results.csv) | 80.8 KB |
| [`results/local_eval/h1_weighted/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_weighted/results.csv) | 80.6 KB |
| [`results/local_eval/h1_mean_transfer+cd4/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_mean_transfer%2Bcd4/results.csv) | 80.5 KB |
| [`results/local_eval/h1_v5eq_exact/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v5eq_exact/results.csv) | 80.5 KB |
| [`results/local_eval/h1_v5eq_exact_de/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v5eq_exact_de/results.csv) | 80.5 KB |
| [`results/local_eval/h1_v6cand/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v6cand/results.csv) | 80.2 KB |
| [`results/local_eval/h1_v6cand_de/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v6cand_de/results.csv) | 80.2 KB |
| [`results/local_eval/h1_v6floor02/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v6floor02/results.csv) | 80.1 KB |
| [`results/local_eval/h1_v6floor01/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/h1_v6floor01/results.csv) | 80.1 KB |
| [`results/local_eval/hepg2_calibrated/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/hepg2_calibrated/results.csv) | 79.6 KB |
| [`results/local_eval/jurkat_norm_restored/results.csv`](https://github.com/MMatinGerami/zero-shot-perturbation/raw/HEAD/results/local_eval/jurkat_norm_restored/results.csv) | 79.6 KB |

> 省略规则:普通源码 >2 MB、文档/构建配置 >1 MB(README 与 .ipynb 不设限),或后缀不在白名单(图片/权重/数据表/二进制/通用 .txt/.log/.csv/.json/.xml 等)、以及子模块。需要全文时按上表链接直接取回,或用 git 克隆上游仓库。

> 本清单由 `scripts/archive.py` 每次归档时自动生成并覆盖。
