# Predicting chromosome flexibility from the genomic sequence based on deep learning neural networks
## How to set up?
1) Install:
   Python 3.6, Tensorflow 1.14.0, Keras 2.2.4, pycharm

2) Clone this repository

3) Untar the downloaded file

## All process
1. Run `data_process.py`
2. Run `train.py`
3. Run `predict.py`
4. Run `get_inf.py`
5. Run `evaluation.py`
6. Run `attention_peak.py`

## Data set
The data set of GM12878 cell line is saved in data/data_set/GM12878/, which contains "neg_data_sequence.txt" and "pos_data_sequence.txt". The "neg_data_sequence.txt" contains the negative samples, and "pos_data_sequence.txt" contains the postive samples.

These file include:
> **start** : Sequence start position,\
> **end** : Sequence end position,\
> **atac** : ATAC value,\
> **dnase** : DNase value,\
> **fri** : FRI value,\
> **gnm** : GNM value,\
> **std(fri)** : Standardized FRI,\
> **std(gnm)** : Standardized GNM,\
> **chr** : The chromosome,\
> **peak_labels** : Whether there is a peak signal,\
> **rank_labels** : ATAC value and DNase value exceed or below the threshold,\
> **labels** : High chromosome flexibility is 1, and low chromosome flexibility is 0,\
> **seq** : Genomic sequence

## Data process
Run "data_process.py" for data processing. The script will do one-hot encode and divide the data set. The "--cell_line" is the cell line of the dataset.
```
python data_process.py
  --cell_line="GM12878"
```

## Training
Run "train.py" to train the model. The model will be saved when the accuracy of validation set is maximum. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3" and "deepcfp".
```
python train.py
  --cell_line="GM12878"
  --model_name="deepcfp"
```

## Predicting
Run "predict.py" to get the results of prediction. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3" and "deepcfp".
```
python predict.py
  --cell_line="GM12878"
  --model_name="deepcfp"
```

## Evaluation
First, run "get_inf.py" to get the results for comparing. The data will be saved in the folder "compare".
```
python get_inf.py
  --cell_line="GM12878"
  --model_name="deepcfp"
```
And then, run "evaluation.py" to get the curves and the bars. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3" and "deepcfp". The "--curve" is the curve's name, which can be selected from "ROC" and "P-R".
```
python evaluation.py
  --cell_line="GM12878"
  --model_name="deepcfp"
  --curve="ROC"
```

## Attention and TF binding site experiment
Run "attention_peak.py", you can get the Figure of the proportion of peak signals in different attention weights. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3" and "deepcfp".
```
python attention_peak.py
  --cell_line="GM12878"
  --model_name="deepcfp"
```


