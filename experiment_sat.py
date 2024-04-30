import pandas as pd
import pyterrier as pt
import torch

if not pt.started():
    pt.init()

torch.cuda.empty_cache()

import pyt_splade

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

factory_log = pyt_splade.SpladeFactory(device='cuda', saturation_function='log')
factory_sigmoid = pyt_splade.SpladeFactory(device='cuda', saturation_function='sigmoid')
factory_squared_inverse = pyt_splade.SpladeFactory(device='cuda', saturation_function='squared_inverse')

# Define a function to create an indexing and retrieval pipeline
def create_pipeline(factory):
    doc_encoder = factory.indexing()
    pt_index_path = './vaswani_' + factory.saturation_function

    indexer = pt.index.IterDictIndexer(pt_index_path, overwrite=True, blocks=True)
    indexer.setProperty("termpipelines", "")
    indexer.setProperty("tokeniser", "WhitespaceTokeniser")

    # Index the documents
    indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
    index_ref = indxr_pipe.index(pt.get_dataset('vaswani').get_corpus_iter(), batch_size=64)

    # Create retrieval pipelines
    br = pt.BatchRetrieve(pt_index_path, wmodel='Tf', verbose=True)
    query_pipeline = factory.query()
    retr_pipeline = query_pipeline >> br

    return retr_pipeline

# Create pipelines for each saturation function
retr_pipe_log = create_pipeline(factory_log)
retr_pipe_sigmoid = create_pipeline(factory_sigmoid)
retr_pipe_squared_inverse = create_pipeline(factory_squared_inverse)

# Load the dataset's topics and qrels
dataset = pt.get_dataset('vaswani')
topics = dataset.get_topics()
qrels = dataset.get_qrels()

# Run the experiment with all three SPLADE configurations
from pyterrier.measures import RR, nDCG, AP
experiment = pt.Experiment(
    [retr_pipe_log, retr_pipe_sigmoid, retr_pipe_squared_inverse],
    topics,
    qrels,
    batch_size=200,
    filter_by_qrels=True,
    eval_metrics=[RR(rel=2), nDCG@10, nDCG@100, AP(rel=2)],
    names=['splade_log', 'splade_sigmoid', 'splade_squared_inverse']
)

print(experiment)
