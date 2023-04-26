import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# import transformers

__all__ = ["tokenize"]
_tokenizer = _Tokenizer()

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# frozen_tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased", TOKENIZERS_PARALLELISM=False)

def Frozen_tokenize(text):
    res = frozen_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    token = res['input_ids']
    return token


def CLIP_tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


tokenize = CLIP_tokenize