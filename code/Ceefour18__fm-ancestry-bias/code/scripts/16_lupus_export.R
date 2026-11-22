suppressMessages({library(Seurat); library(Matrix)})
obj <- readRDS("data/raw/GSE189050/GSE189050_final_seurat.RDS")
DefaultAssay(obj) <- "RNA"
cnt <- tryCatch(GetAssayData(obj, assay = "RNA", layer = "counts"),
                error = function(e) GetAssayData(obj, assay = "RNA", slot = "counts"))
cat("Max count:", max(cnt), "| all whole numbers:", all(cnt@x == round(cnt@x)), "\n")
out <- "data/processed/lupus"
writeMM(cnt, file.path(out, "counts.mtx"))
writeLines(rownames(cnt), file.path(out, "genes.txt"))
writeLines(colnames(cnt), file.path(out, "barcodes.txt"))
keep <- c("run","subject_id","ancestry","classification","age","nCount_RNA","nFeature_RNA",
          "percent_mt","coarse_cell_type","fine_cell_type","clusters_annotated")
write.csv(obj@meta.data[, keep], file.path(out, "obs.csv"))
cat("Exported", ncol(cnt), "cells\n")
