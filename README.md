# RNA Nucleotides Embeddings

Deeply learning 3D-aware node embeddings for RNA secondary structure graphs. 

## Pretraining using context prediction 

Learn unsupervised node embeddings using method from https://arxiv.org/abs/1905.12265

To train model on RNA graphs in ./directory:

First preprocess graphs (compute angles and one-hot features from the rna_classes features) by 
```
python data_processing/preprocess_graphs.py -i [directory] -o [preprocessed_dir]
```

Then learn embeddings using context prediction by running 
```
python train.py --train_dir [preprocessed_dir] 
```

Default values for context prediction hyperparams are K=1, r1 = 1, r2=3.
(K,r1,r2) can be changed by adding arguments 
```
python train.py --train_dir [preprocessed_dir] --K ... --r1 ... --r2 ...
```

## Annotating graphs 

To compute embeddings for graphs in ./directory and save them to a new dir, run
```
python embeddings.py -i [directory] -o [write_directory]
``` 

## Visualize learned embeddings for different basepair types 

To view embeddings in PCA space, colored by true basepair type, run :
```
python post/visualize_embeddings.py 
```
