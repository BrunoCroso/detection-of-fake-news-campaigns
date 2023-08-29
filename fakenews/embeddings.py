from typing import Dict, List, Union

import numpy as np

import models
import utils

from transformers import AutoModel, AutoTokenizer
import torch

class UserEmbedder:
    def __init__(self, bertweet_model, not_in_vocabulary_embedding: Union[List, np.ndarray]=None):
        self.__bertweet_model = bertweet_model
        self.__embedding_dimensions = 768
        self.__tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
        if not_in_vocabulary_embedding is None:
            self.__not_in_vocabulary_embedding = np.zeros(self.__embedding_dimensions)
        else:
            self.__not_in_vocabulary_embedding = not_in_vocabulary_embedding

    def embed(self, user: models.User):
        embedding = np.array([])
        if self.__bertweet_model is not None:

            tokens = torch.tensor([self.__tokenizer.encode(user.description, truncation=True)])
            with torch.no_grad():
                word_embeddings = self.__bertweet_model(tokens)[0][0]

            if len(word_embeddings) != 0:
                embedding = torch.mean(word_embeddings, axis=0).numpy()
            else:
                embedding = self.__not_in_vocabulary_embedding
        return embedding
