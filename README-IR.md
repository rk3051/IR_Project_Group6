# pytsplade-IR-Group6

## How to install and run the system:

1. Install all dependencies using:
```
    pip install -r /pytsplade-master/requirements.txt
```
2. Run setup.py to initialize the system with the following args / flags:
```    
    python3 setup.py install --user
```

3. To run our experiment, run the following command
```
    python3 experiment.py --sigmoid
```

You can choose any of the following flags in place of --sigmoid to choose which saturation function you'd like to run. (Running it with no flag, defaults to the logarithmic saturation function):

- logarithmic: `--log`
- sigmoid: `--sigmoid`
- square root: `--sqrt`
- no saturation function: `--none`
- tanh: `--tanh`
- log2: `--log2`

## Changes made in the codebase, and which files were modified 

1. Created requirements.txt to easily install all required dependencies

2. Modified `pyt_splade/__init__.py` (the file which initializes PyTerrier Splade) to accept our modified SPLADE model, as well as the `saturation_function` argument we introduced, which enables configuration of the saturation function used. 

```
from .transformer_rep import Splade

class SpladeFactory():

    def __init__(
        self,
        model : Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil",
        tokenizer=None,
        agg='max',
        max_length = 256,
        device=None,
        saturation_function='sigmoid'):
```

Here our modifications allow us to instantiate our custom SPLADE model.

```
        if isinstance(model, str):
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = Splade(model, agg=agg, saturation_function=self.saturation_function)
            self.model.eval()
            self.model = self.model.to(self.device)
```

3. Modified `transformer_rep` from the SPLADE repository (as well as migrating its dependencies to our project) to create our modified SPLADE model. These modifications included, being able to configure the saturation function, as well as creating the saturation functions. 

```
class Splade(SiameseBase):
    """SPLADE model
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None, freeze_d_model=False, agg="max", fp16=True, **kwargs):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output="MLM",
                         match="dot_product",
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,
                         fp16=fp16)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size  # output dim = vocab size = 30522 for BERT
        assert agg in ("sum", "max")
        self.agg = agg
        self.saturation_function = kwargs.get('saturation_function', 'log')

    # Our saturation functions
    def apply_saturation(self, logits, attention_mask):
        # Apply different saturation functions based on the configuration
        if self.saturation_function == 'log':
            return torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        elif self.saturation_function == 'sigmoid':
            # Sigmoid with fixed bounds (example bounds; adjust as necessary)
            return torch.sigmoid(torch.relu(logits)) * attention_mask.unsqueeze(-1) - 0.5
        elif self.saturation_function == 'sqrt':
            # Squared inverse saturation
            saturated_logits = torch.sqrt(torch.relu(logits)) * attention_mask.unsqueeze(-1)
            return saturated_logits
        elif self.saturation_function == 'none':
            return torch.relu(logits) * attention_mask.unsqueeze(-1)
        elif self.saturation_function == 'tanh':
            return torch.tanh(torch.relu(logits)) * attention_mask.unsqueeze(-1)
        elif self.saturation_function == "log2":
            return torch.log2(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        else:
            raise ValueError("Unsupported saturation function")
```

4. Created `experiment.py` which performs indexing and retrieval experiments using PyTerrier along with our custom SPLADE models, leveraging the Vaswani Dataset. 