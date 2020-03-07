# DeepFRED

Deeply learning 3D-aware node embeddings for RNA secondary structure graphs. 

## Pretraining using context prediction 

https://arxiv.org/abs/1905.12265


To train model on RNA graphs in ./directory, run
```
python train.py --train_dir [directory]
```

## Annotating graphs 

To compute embeddings for graphs in ./directory and save them to a new dir, run
```
python eval/embeddings.py -i [directory] -o [write_directory]
``` 

## Visualize learned embeddings for different basepair types 

(TODO)
```
python eval/visualize_embeddings.py 
```
