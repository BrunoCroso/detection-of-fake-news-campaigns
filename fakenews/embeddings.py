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

    
    def embed(self, users): #Tem que poder receber n users e retornar os respectivos n embeddings (recebe uma lista com users)
        embedding = np.array([])
        if self.__bertweet_model is not None:
            descriptions = []
            for user in users:
                if 'description' in user:
                    descriptions.append(user['description'])
                else:
                    descriptions.append('')

            encoded_sentences = self.__tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.__bertweet_model(**encoded_sentences)

            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

        return sentence_embeddings
