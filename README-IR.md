# pytsplade-IR-Group6

1. How to install and run the system:

i. Install all dependencies using:
```
    pip install -r /pytsplade-master/requirements.txt
```
ii. Run setup.py to initialize the system with the following args / flags:
```    
    python3 setup.py install --user
```

2. Changes made in the codebase, and which files were modified 

Created requirements.txt to easily install all required dependencies

To support our investigation into the impact of term importance prediction strategies on retrieval, we modified pytsplade-master/pyt_splade/__init__.py to introduce new saturation functions and adjustable L-FLOPS regularization, aiming to explore how these changes affect the SPLADE model's effectiveness and efficiency in information retrieval tasks.