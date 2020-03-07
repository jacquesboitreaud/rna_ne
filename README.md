# DeepFRED

Deeply learning 3D-aware node embeddings for RNA secondary structure graphs. 

## Pretraining using context prediction 

https://arxiv.org/abs/1905.12265


To train model on RNA graphs in ./directory:

First preprocess graphs (compute angles and one-hot features from the rna_classes features) by 
```
python data_processing/preprocess_graphs.py -i [directory] -o [preprocessed_dir]
```

Then learn embeddings using context prediction by running 
```
python train.py --train_dir [preprocessed_dir]
```

## Annotating graphs 

To compute embeddings for graphs in ./directory and save them to a new dir, run
```
python embeddings.py -i [directory] -o [write_directory]
``` 

## Visualize learned embeddings for different basepair types 

(TODO)
```
python eval/visualize_embeddings.py 
```
