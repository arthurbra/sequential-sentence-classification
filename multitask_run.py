import time
import gc
from os import makedirs
from datetime import datetime
from eval_run import eval_and_save_metrics
from gpu_executor import GPUExecutor
from utils import log, get_device, ResultWriter
from task import pubmed_task, nicta_task, NICTA_LABELS, PUBMED_LABELS, dri_task, art_task, PUBMED_TASK, NICTA_TASK, \
    DRI_TASK, ART_TASK, pubmed_task_small, art_task_small, generic_task, GEN_ART_TASK, GEN_NIC_TASK, GEN_PMD_TASK, \
    GEN_DRI_TASK

from train import SentenceClassificationTrainer, SentenceClassificationMultitaskTrainer
from models import BertHSLN


# BERT_VOCAB = "bert-base-uncased"
# BERT_MODEL = "bert-base-uncased"
# BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
BERT_MODEL = "bert_model/scibert_scivocab_uncased/"


config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "cacheable_tasks": [],

    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,


    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size":  32,
    "max_seq_length": 128,
    "max_epochs": 20,

    # set true only for generic datasets
    "generic_output_layer": True

}


MAX_DOCS = 1
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


def create_generic_task(task_name):
    return generic_task(task_name, train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


# ADAPT: uncomment the tasks you want to be included in the training
def get_tasks():
    tasks = []
    tasks.append(create_task(pubmed_task))
    #tasks.append(create_task(pubmed_task_small))
    tasks.append(create_task(nicta_task))
    tasks.append(create_task(dri_task))
    tasks.append(create_task(art_task))
    #tasks.append(create_task(art_task_small))

    #tasks.append(create_generic_task(GEN_DRI_TASK))
    #tasks.append(create_generic_task(GEN_PMD_TASK))
    #tasks.append(create_generic_task(GEN_NIC_TASK))
    #tasks.append(create_generic_task(GEN_ART_TASK))

    return tasks

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

# ADAPT: adapt the folder name of the run if necessary
run = f"{timestamp}_mult_all"

# ADAPT: provide the GPU numbers. If several are provided then the trainings per fold resp. restart  will be executed concurrently.
gpus = [0]

# ADAPT: Set to False if you do not want to save the best model. All models for each epoch are saved during the training.
save_models = True
# -------------------------------------------


run_results = f'results/{run}'
makedirs(run_results, exist_ok=False)

# determine task with maximum number of folds
task_with_max_fold = get_tasks()[0]
for task in get_tasks():
    if task.num_folds > task_with_max_fold.num_folds:
        task_with_max_fold = task

log(f"Task with maximum number of folds: " + task_with_max_fold.task_name)
log(f"GPUs: {gpus}")

executor = GPUExecutor(gpus)

def run_training(restart, fold_num):
    def run(gpu):
        tasks = get_tasks()
        # preload data if not already done
        for task in tasks:
            log(f"preloading folds for task {task.task_name}")
            task.get_folds()

        start = time.time()
        result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} of {task_with_max_fold.num_folds}")
        result_writer.log(f"Starting training {restart} for fold {fold_num}... ")

        train_batches = []
        dev_batches = []
        test_batches = []
        for task in tasks:
            # determine the folds of other tasks
            task_fold = task.get_folds()[fold_num % task.num_folds]
            train_batches.extend(task_fold.train)
            dev_batches.extend(task_fold.dev)
            test_batches.extend(task_fold.test)

        trainer = SentenceClassificationMultitaskTrainer(get_device(gpu), config, tasks, result_writer)
        trainer.run_training(train_batches, dev_batches, test_batches, restart, fold_num,
                             save_models=save_models, save_best_model_path=run_results)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")

        # explicitly call garbage collector so that CUDA memory is released
        gc.collect()
    return run

restarts = 3 if task_with_max_fold.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num in range(task_with_max_fold.num_folds):
        executor.submit(run_training(restart, fold_num))

executor.shutdown()

log("Training finished.")

log("Calculating metrics...")
eval_and_save_metrics(run_results)
log("Calculating metrics finished")


