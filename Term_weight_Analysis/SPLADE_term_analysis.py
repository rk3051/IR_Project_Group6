from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd
import matplotlib.pyplot as plt

"""
    This script analyzes importance estimation of single documents given
    different saturation functions._
"""

def transform(output, tokens, saturation="log"):
    if saturation == "log":
        vec = torch.sum(
            torch.log(
                1 + torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
        
        # print((torch.log(
        #         1 + torch.relu(output.logits)
        #     ) * tokens.attention_mask.unsqueeze(-1)))
        #exit()
    elif saturation == "log2":
        vec = torch.sum(
            torch.log2(
                1 + torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
        
    elif saturation == "custom":
        
        squared_inverse = 1 - (1 / (1 + torch.square(torch.relu(output.logits))))
        
        # Apply saturation behavior
        scaled_result = 5 * squared_inverse * (1 - torch.exp(-output.logits / 5))
        
        vec = torch.sum(
            scaled_result * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
        
    elif saturation == "sqrt":
        vec = torch.sum(
            torch.sqrt(
                torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
    elif saturation == "sigmoid":
        vec = torch.sum(
            torch.sigmoid(
                torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1) - 0.5,
        dim=1)[0].squeeze()
    elif saturation == "squared":
        vec = torch.sum(
            torch.square(
                torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
    elif saturation == "tanh":
        vec = torch.sum(
            torch.tanh(
                torch.relu(output.logits)
            ) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
    elif saturation == "none":
        vec = torch.sum(
                torch.relu(output.logits) * tokens.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze()
    return vec
        


def main():
    model_id = 'naver/splade-cocondenser-ensembledistil'
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    
    # doc1 from vaswani
    text = (
            "the british computer society  report of a conference held in cambridge june"
    )
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    
    # check the shape the output logits
    # print(output.logits.shape)
    # exit()

    #print(output.logits.shape)
    # tokens['input_ids] # shows ids of each token
    
    # # Convert token IDs to their corresponding text tokens
    # token_ids = tokens['input_ids'].squeeze()  # Remove batch dimension if only one example
    # tokens_text = tokenizer.convert_ids_to_tokens(token_ids)

    # # Print each token ID with its corresponding text representation
    # for token_id, token_text in zip(token_ids, tokens_text):
    #     print(f'{token_id} --> {token_text}')
        
    # exit()
    
    # shows all functions implemented
    # saturation_functs = ["log", "sq_inverse", "sigmoid", "none", "sqrt", "tanh", "log2"]
    
    saturation_functs = ["log", "none", "sqrt", "log2", "sigmoid", "tanh"]
    results = {}
    
    # Take output logits and transform them into a sparse vector
    for saturation in saturation_functs:
        vec = transform(output, tokens, saturation)
        idx2token = {
            idx: token for token, idx in tokenizer.get_vocab().items()
        }
        # Extract non-zero values
        cols = vec.nonzero().squeeze().cpu().tolist()
        # print(len(cols), len(vec))
    
        weights = vec[cols].cpu().tolist()
        # use to create a dictionary of token ID to weight
        sparse_dict = dict(zip(cols, weights))
    
        #  map token IDs to human-readable tokens
        sparse_dict_tokens = {
            idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }
        # sort so we can see most relevant tokens first
        sparse_dict_tokens =  {
            k: v for k, v in sorted(
                sparse_dict_tokens.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        results[saturation] = sparse_dict_tokens
    
    df = pd.DataFrame.from_dict(results, orient='index').transpose()
    
    #df.to_csv("saturation_results.csv") # create csv of term weights
    
    # Sort the DataFrame based on 'log' weights from highest to lowest
    df_sorted = df.sort_values(by='log', ascending=False)
    
    # Plotting all saturation functions on the same graph, following the order of 'log'
    plt.figure(figsize=(14, 7))
    for column in df_sorted.columns:
        plt.plot(df_sorted[column], label=column, marker='o')
    
    plt.title('Comparison of Term Weights Across Saturation Functions')
    plt.xlabel('Terms')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.legend(title='Saturation Type')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        


if __name__ == "__main__":
    main()