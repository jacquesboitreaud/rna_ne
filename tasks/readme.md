# Scripts for downstream learning tasks on RNA annotated 2D graphs

## Magnesium binding site prediction

Find magnesium binding sites in RNA graphs: node-level task (nucleotide classification)

Binding sites for different distance thresholds are extracted and stored in pickle file in ../data
To add 'Mg_binding node feature' to networkx rna graphs, run 

```
python build_mg_dataset.py -i [graphs directory] -c [cutoff binding site distance, in A] 
```
Annotated graphs will be saved to ../data/mg_graphs
