import pandas as pd
import pyterrier as pt
import torch
import argparse

parser = argparse.ArgumentParser(description='Run the document indexing and retrieval experiment with different saturation functions.')

parser.add_argument('--log', action='store_true', help='Use logarithmic saturation function')
parser.add_argument('--sigmoid', action='store_true', help='Use sigmoid saturation function')
parser.add_argument('--sqrt', action='store_true', help='Use square root saturation function')
parser.add_argument('--none', action='store_true', help='Use no saturation function')
parser.add_argument('--tanh', action='store_true', help='Use tanh saturation function')
parser.add_argument('--log2', action='store_true', help='Use log2 saturation function')

args = parser.parse_args()

if args.log:
    sat_func = 'log'
elif args.sigmoid:
    sat_func = 'sigmoid'
elif args.sqrt:
    sat_func = 'sqrt'
elif args.none:
    sat_func = 'none'
elif args.tanh:
    sat_func = 'tanh'
elif args.log2:
    sat_func = 'log2'
else:
    # Default to log if no flag is set
    sat_func = 'log'

if not pt.started():
    pt.init()

torch.cuda.empty_cache()

import pyt_splade

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# in factory, change field `saturation_function`
factory = pyt_splade.SpladeFactory(device='cuda', saturation_function=sat_func)
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
    eval_metrics=[MRR(rel=1), nDCG@10, nDCG@100, MAP(rel=1)],
    names=[f'splade_{sat_func}', 'bm25', 'tfidf']
)

print(experiment)
