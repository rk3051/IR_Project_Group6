import pandas as pd
import pyterrier as pt
import torch


if not pt.started():
    pt.init()

torch.cuda.empty_cache()

import pyt_splade

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


factory = pyt_splade.SpladeFactory(device='cpu', saturation_function='squared_inverse')
doc_encoder = factory.indexing()
pt_index_path = './vaswani'

dataset = pt.get_dataset('vaswani')
indexer = pt.index.IterDictIndexer(pt_index_path, overwrite=True, blocks=True)
indexer.setProperty("termpipelines", "")
indexer.setProperty("tokeniser", "WhitespaceTokeniser")

indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
index_ref = indxr_pipe.index(dataset.get_corpus_iter(), batch_size=64)

br = pt.BatchRetrieve(pt_index_path, wmodel='Tf', verbose=True)
query_splade = factory.query()
retr_pipe = query_splade >> br

br_bm25 = pt.BatchRetrieve(pt_index_path, wmodel='BM25', verbose=True)
br_tfidf = pt.BatchRetrieve(pt_index_path, wmodel='TF_IDF', verbose=True)

# Define the dataset's topics and qrels
topics = dataset.get_topics()
qrels = dataset.get_qrels()

# Run the experiment with SPLADE, BM25, and TF-IDF
from pyterrier.measures import *
experiment = pt.Experiment(
    [retr_pipe, br_bm25, br_tfidf],
    topics,
    qrels,
    batch_size=200,
    filter_by_qrels=True,
    eval_metrics=[RR(rel=1), nDCG@10, nDCG@100, AP(rel=1)],
    names=['splade_squared_inverse', 'bm25', 'tfidf']
)

print(experiment)
