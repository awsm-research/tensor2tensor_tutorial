# Tensor2Tensor Tutorial for training and evaluating data from text file

## Tensor2Tensor Installation
Please see tensor2tensor_installation.md for more detail.

## How to register your own problem with tensor2tensor
1. Create directory to store python file. The directory name should contain words that are separated by '_'. For example, my_problem .
2. In the created directory, create python file to store configuration of the problem and parameter setting. The python file name must be camel-case style and should have the same words as directory name and ends with '_problem. For example, `MyProblem_problem.py`
3. In the created directory, create `__init__.py`. In this created file, add this import statement `from .MyProblem_problem import *`. This import statement is actually a python file name in step 2.
4. Please see more detail in `my_problem/MyProblem_problem.py`

Note: you can change `my_problem` and `MyProblem` to other names that you want.

## How to register your own hyperparameter with tensor2tensor
1. Go to the python file that stores problem configuration (created at step 2 of **How to register your own problem with tensor2tensor** section.
2. In this file, add the following code 

	    @registry.register_hparams
	    def TransformerHparams1():
		   hparams = transformer.transformer_base_single_gpu() # this value can be changed
	       hparams.batch_size = 5000 # this value can be changed
	       hparams.num_encoder_layers = 1 # this value can be changed
	       hparams.num_decoder_layers = 2 # this value can be changed
	       hparams.hidden_size = 256 # this value can be changed
	       hparams.num_heads = 8 # this value can be changed
	       hparams.eval_drop_long_sequences = True # this value can be changed
	       hparams.max_length = 1200 # this value can be changed
	       return hparams
For more detail of parameters, please see `tensor2tensor/models/transformer.py
`of tensor2tensor repository.
Note: the function name must be camel-case style and must not contain underscore ('_').
