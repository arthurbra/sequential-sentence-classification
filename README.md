# Multi-Task Learning for Sequential Sentence Classification
This repository contains the source code for the paper

_Arthur Brack, Anett Hoppe, Pascal Buschermöhle, Ralph Ewerth (2021): Cross-Domain Multi-Task Learning for Sequential Sentence Classification in Research Papers_
 

# Installation
Create an anaconda environment "scibert" from the file environment.yaml with the following command:

- conda env create -f environment.yaml

You can activate the environment with the command "conda activate scibert"

Note: the environemnt has been created with "conda env export --no-builds > environment.yaml"

# Dataset preparation

## SciBERT
- Download SciBERT from https://github.com/allenai/scibert (Pytorch HuggingFace, scibert-scivocab-uncased )
- extract all files into the directory "bert_model/scibert_scivocab_uncased/"

## Download and Conversion of corpora

### PUBMED-20k and NICTA-PIBOSO

- download the files from https://github.com/jind11/HSLN-Joint-Sentence-Classification/tree/master/data
- put the files of Pubmed-20k into the folder  "datasets/pubmed-20k"
- put the files of NICTA-PIBOSO into the folder "datasets/nicta-piboso"

### Dr. Inventor
- obtain the corpus as described in http://sempub.taln.upf.edu/dricorpus
- create a folder "datasets/DRI"
- convert the XML-files into a text file with convert_dri.py into the folder "datasets/DRI"

### ART
- download the corpus from https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/
- convert the XML-files into a text file with convert_dri.py into the folder "datasets/ART"

### Generic dataset
- compile the dataset with compile_generic_dataset.py

## SciBERT tokenisation

To speed up the training we tokenize the sentences with SciBERT beforehand.

- Just execute tokenize_files.py
- You should get *_scibert.txt files in the resp. dataset folders



# Experiments

- All experiment results of a run are stored in a separate folder under "results/".
- You can choose in the run scripts whether model files shall be saved. 
- We use simple python scripts instead of command-line arguments or shell scripts for simple customising of the training runs.

After each run different metrics are calculated and stored in the results folder:

- results.csv (contains for each task averaged metrics with the best dev metric in the resp. fold/restart)
- f1_per_label.csv (contains averaged F1 scores for each task)
- *_cm.pdf (Confusion matrix for each task)

## Baseline Results

To execute the baseline experiments do the following:

- open the file baseline_run.py
- go through the "# ADAPT:" comments and adapt the settings
- execute the script
- the script trains the model for each fold resp. restart and evaluates their performance

## Sequential Transfer Learning (INIT)

- train first the baseline models and let them be saved (save_best_models=True)
- open the file transfer_run.py
- go through the "# ADAPT:" comments and adapt e.g. the source task and the target task you want to execute
- execute the script
- the script will load the baseline models as the source task and train the target task, and then evaluate the performance

## Multi-Task Learning (MULT ALL)
During Multi-Task Learning you can provide several GPUs (see gpus variable) so that the runs can be executed concurrently.

- open the file multitask_run.py
- go through the "# ADAPT:" comments and adapt everything you need (e.g. the tasks to be included)
- execute the script
- the script will train and evaluate the models

## Multi-Task Learning with grouped Layers (MULT GROUPED)
During Multi-Task Learning you can provide several GPUs (see gpus variable) so that the runs can be executed concurrently.
Please note that each fold resp. restart two GPUs are required as the model does not fit into the memory of a 12 GB RAM GPU.

- open the file multitask_run_sep_layers.py
- go through the "# ADAPT:" comments and adapt everything you need 
- execute the script
- the script will train and evaluate the models

# Semantic Relatedness of Labels
To calculate the semantic relatedness between the datasets for different models you first need the following:

- baseline models for all tasks
- Multi-Task model (MULT ALL)
- Multi-Task model with grouped layers (MULT GROUPED ((P,N),(D,A))) 

Then open the jupyter notebook semantic_relatedness_labels.ipynb

- adapt the paths to the models (see "# ADAPT:" comments)
- adapt the result  (see "# ADAPT:" comments)
- run the jupyter notebook
- in the result folder, you should find semantic vectors, semantic vectors as heatmaps and PCA reductions.

To calculate the Silhouette scores:

- open the the jupyter notebook 'silhouette_analysis_label_clusters.ipynb'
- adapt the paths  (see "# ADAPT:" comments)
- run the jupyter notebook


