"""Stage 1b node: map zebrafish gene counts to human genes."""

from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class OrthologMapNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_ortholog_map",
        stage="analysis",
        tool="AnnData",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Convert a zebrafish count matrix to human genes with an Ensembl ortholog table. A zebrafish gene is kept if it has exactly one human ortholog; counts of zebrafish genes that share a human gene are summed; all other genes are dropped.",
        parameters={
            "table_path": ParameterDefinition(default_value="data/raw/ensembl_orthologs/zebrafish_human_orthologs.tsv", description="Tab-separated BioMart table with a header: zebrafish gene id, human gene id, human gene name, homology type."),
            "out_path": ParameterDefinition(default_value="data/derived/atlas_human_genes.h5ad", description="File to write."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file of raw counts with zebrafish Ensembl gene ids as the var index."),
        },
        outputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file of summed raw counts with human Ensembl gene ids as the var index, a var column 'symbol', and the obs table unchanged."),
            "mapping": PortDefinition(type=PortType.DICT, description="Zebrafish genes in, zebrafish genes kept, human genes out, and the share of all counts that survive."),
        },
        methods={
            "map": MethodDefinition(description="Apply the ortholog rule and sum counts.", inputs=["data_path"], outputs=["data_path", "mapping"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("map", self.map, method_key="map")

    def map(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        adata = ad.read_h5ad(data_path)
        table = pd.read_csv(p["table_path"], sep="\t", header=0, names=["zebrafish", "human", "symbol", "kind"], dtype=str).dropna(subset=["human"])
        table = table[table["human"].str.startswith("ENSG")].drop_duplicates(["zebrafish", "human"])
        single = table.groupby("zebrafish")["human"].transform("nunique") == 1
        table = table[single].set_index("zebrafish")
        kept = [g for g in adata.var_names if g in table.index]
        humans = sorted(set(table.loc[kept, "human"]))
        column = {h: i for i, h in enumerate(humans)}
        rows = np.array([adata.var_names.get_loc(g) for g in kept])
        cols = np.array([column[table.at[g, "human"]] for g in kept])
        collapse = sp.csr_matrix((np.ones(len(kept), dtype=np.float32), (rows, cols)), shape=(adata.n_vars, len(humans)))
        counts = sp.csr_matrix(adata.X)
        mapped = (counts @ collapse).tocsr()
        symbol = table.drop_duplicates("human").set_index("human")["symbol"].reindex(humans).fillna("")
        out_adata = ad.AnnData(X=mapped, obs=adata.obs.copy(), var=pd.DataFrame({"symbol": symbol.to_numpy()}, index=humans))
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out_adata.write_h5ad(out)
        mapping = {
            "zebrafish_genes_in": int(adata.n_vars),
            "zebrafish_genes_kept": int(len(kept)),
            "human_genes_out": int(len(humans)),
            "share_of_counts_kept": round(float(mapped.sum() / counts.sum()), 4),
        }
        return {"data_path": str(out), "mapping": mapping}
