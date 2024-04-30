import pyterrier as pt
assert pt.started()
from typing import Union
import torch
import pandas as pd
from .transformer_rep import Splade

def _matchop(t, w):
    import base64
    import string
    if not all(a in string.ascii_letters + string.digits for a in t):
        encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8") 
        t = f'#base64({encoded})'
    if w != 1:
        t = f'#combine:0={w}({t})'
    return t

class SpladeFactory():

    def __init__(
        self,
        model : Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil",
        tokenizer=None,
        agg='max',
        max_length = 256,
        device=None,
        saturation_function='sigmoid'):
        
        import torch
        self.max_length = max_length
        self.model = model
        self.tokenizer = tokenizer
        self.saturation_function = saturation_function

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        if isinstance(model, str):
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = Splade(model, agg=agg, saturation_function=self.saturation_function)
            self.model.eval()
            self.model = self.model.to(self.device)
        elif isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
            if self.tokenizer is None:
                raise ValueError("You must specify tokenizer if passing a model")

        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def indexing(self) -> pt.Transformer:
        def _transform_indexing(df):
            rtr = []
            if len(df) > 0:
                with torch.no_grad():
                    # now compute the document representation
                    doc_reps = self.model(d_kwargs=self.tokenizer(
                        df.text.tolist(),
                        add_special_tokens=True,
                        padding="longest",  # pad to max sequence length in batch
                        truncation="longest_first",  # truncates to max model length,
                        max_length=self.max_length,
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(self.device))["d_rep"]  # (sparse) doc rep in voc space, shape (docs, 30522,)

                    for i in range(doc_reps.shape[0]): #for each doc
                        # get the number of non-zero dimensions in the rep:
                        col = torch.nonzero(doc_reps[i]).squeeze().cpu().tolist()

                        # now let's create the bow representation as a dictionary               
                        weights = doc_reps[i,col].cpu().tolist()
                        d = {self.reverse_voc[k] : v for k, v in sorted(zip(col, weights), key=lambda x: (-x[1], x[0]))}
                        rtr.append(d)
            return df.assign(toks=rtr)
        return pt.apply.generic(_transform_indexing)

    
    def query(self, mult=100) -> pt.Transformer:
    
        def _transform_query(df):
            from pyterrier.model import push_queries
            new_queries = []
            if len(df) > 0:
                with torch.no_grad():
                    # now compute the query representations
                    query_reps = self.model(q_kwargs=self.tokenizer(
                        df['query'].tolist(),
                        add_special_tokens=True,
                        padding="longest",  # pad to max sequence length in batch
                        truncation="longest_first",  # truncates to max model length,
                        max_length=self.max_length,
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(self.device))["q_rep"]  # (sparse) q rep in voc space, shape (queries, 30522,)
                
                    for i in range(query_reps.shape[0]): #for each query
                        # get the number of non-zero dimensions in the rep:
                        cols = torch.nonzero(query_reps[i]).squeeze().cpu().tolist()
                        # and corresponding weights               
                        weights = query_reps[i,cols].cpu().tolist()

                        # Now let's create the bow representation in terrier's matchop QL.
                        # We scale by mult(=100) to better match the quantized weights in 
                        # the inverted index created by toks2doc(). These defaults match the
                        # parameters suggested for Anserini in Splade repo, namely
                        # quantization_factor_document=100 quantization_factor_query=100.
                        newquery = ' '.join( _matchop(self.reverse_voc[k], v * mult) for k, v in sorted(zip(cols, weights), key=lambda x: (-x[1], x[0])))
                        new_queries.append(newquery)
            
            rtr = push_queries(df)
            rtr['query'] = new_queries
            return rtr
        return pt.apply.generic(_transform_query)
    

def toks2doc(mult=100):
    
    def _dict_tf2text(tfdict):
        rtr = ""
        for t in tfdict:
            for i in range(int(mult*tfdict[t])):
                rtr += t + " " 
        return rtr

    def _rowtransform(df):
        df = df.copy()
        df["text"] = df['toks'].apply(_dict_tf2text)
        df.drop(columns=['toks'], inplace=True)
        return df
    
    return pt.apply.generic(_rowtransform)

