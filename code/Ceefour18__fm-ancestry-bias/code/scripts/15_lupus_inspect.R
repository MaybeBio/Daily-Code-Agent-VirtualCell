suppressMessages(library(Seurat))
obj <- readRDS("data/raw/GSE189050/GSE189050_final_seurat.RDS")
cat("Class:", class(obj), "\n")
cat("Cells x genes:", ncol(obj), "x", nrow(obj), "\n")
cat("Assays:", paste(Assays(obj), collapse = ", "), "\n\n")
md <- obj@meta.data
cat("Metadata columns:\n"); print(colnames(md))
cat("\nFirst rows:\n"); print(head(md, 3))
for (col in colnames(md)) {
  n <- length(unique(md[[col]]))
  if (n > 1 && n <= 60) { cat("\n==", col, "(", n, "values ) ==\n"); print(table(md[[col]])) }
}
