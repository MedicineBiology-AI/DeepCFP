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

## The file naming format
For convenience, we create "change.py". You can change the cell line name and the model name in "change.py", the file name will contain these information.

## Data set
The data set of GM12878 cell line is saved in data/data_set/GM12878/, which contains "neg_data_sequence_v3.txt" and "pos_data_sequence_v3.txt". The "neg_data_sequence_v3.txt" contains the negative samples, and "pos_data_sequence_v3.txt" contains the postive samples.

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

Folder "cross_validation" in "data" is used to save data set for 10-flod cross validation. Folder "final_set" in data is used to save the data set after data process.

## Data process
Run "data_process.py" for data processing. The script will do one-hot encode and divide the data set.If you want to do 10-flod cross validation,
please comment the function " save_final_data" and uncomment the function "save_cross_validation_data".
```
python data_process.py
```

## Training
Run "train.py" to train the model. The model will be saved when the accuracy of validation set is maximum. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3", "model4" and "deepcfp".
```
python train.py
  --model_name="deepcfp"
```

## Predicting
Run "predict.py" to get the results of prediction. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3", "model4" and "deepcfp".
```
python predict.py
  --model_name="deepcfp"
```

## Evaluation
First, run "get_inf.py" to get the results for comparing. And then, run "evaluation.py" to get the Figure of AUROC and AUPR on test set and each chromosome. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3", "model4" and "deepcfp".
```
python get_inf.py
  --model_name="deepcfp"
```
```
python evaluation.py
  --model_name="deepcfp"
```

## Attention and TF binding site experiment
Run "attention_peak.py", you can get the Figure of the proportion of peak signals in different attention weights. The "--model_name" is the model that needs to be trained, which can be selected from "model1", "model2", "model3", "model4" and "deepcfp".
```
python attention_peak.py
  --model_name="deepcfp"
```


