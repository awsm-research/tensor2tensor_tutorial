import sys, os

import tensorflow as tf
from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor import models
from tensor2tensor import problems


@registry.register_problem
class LearnCodeChange(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
    # We can use different types of vocabularies, `VocabType.CHARACTER`,
    # `VocabType.SUBWORD` and `VocabType.TOKEN`.

        return text_problems.VocabType.TOKEN

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return '<UNK>'

@registry.register_hparams
def TransformerHparams1():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    # hparams.batch_size = 3000
    hparams.num_encoder_layers = 1
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 256
    hparams.num_heads = 8
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams
