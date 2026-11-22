suppressMessages(library(Matrix))
f <- "data/raw/GSE162632/zenodo/inputs/1_calculate_pseudobulk/mergedAllCells_withCellTypeIdents_CLEAN.rds"
obj <- readRDS(f)
cat("Seurat version:", as.character(obj@version), "\n")
cat("Assays:", paste(names(obj@assays), collapse = ", "), "| active:", obj@active.assay, "\n")
for (a in names(obj@assays)) {
  x <- obj@assays[[a]]
  cat(a, "counts:", paste(dim(x@counts), collapse = " x "),
      "| max:", if (length(x@counts@x)) max(x@counts@x) else NA,
      "| whole numbers:", all(head(x@counts@x, 1e6) == round(head(x@counts@x, 1e6))), "\n")
}
md <- obj@meta.data
md$cell_type_ident <- as.character(obj@active.ident)
cat("\nCells:", nrow(md), "\nMetadata columns:\n"); print(colnames(md))
cat("\nFirst rows:\n"); print(head(md, 3))
for (col in colnames(md)) {
  n <- length(unique(md[[col]]))
  if (n > 1 && n <= 100) { cat("\n==", col, "(", n, "values ) ==\n"); print(table(md[[col]])) }
}
dir.create("data/processed/randolph", showWarnings = FALSE)
write.csv(md, "data/processed/randolph/cell_metadata.csv")
cat("\nSaved cell metadata:", nrow(md), "cells\n")
