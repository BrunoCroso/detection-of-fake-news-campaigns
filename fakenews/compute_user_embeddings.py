#!/usr/bin/env python

#
# Compute user profile embeddings.
#

import argparse
import json
import os
import logging

import models 
import embeddings
import utils
import numpy as np

from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm


class UserProfiles: 
    def __init__(
        self,
        user_profiles_path,
        user_embeddings_path,
        embeddings_file,
        users_embeddings_lookup,
        not_in_lookup_embedding
    ):
        self._user_profiles_path = user_profiles_path
        self._user_embeddings_path = user_embeddings_path
        self._embeddings_file = embeddings_file
        self._users_embeddings_lookup = users_embeddings_lookup
        self._not_in_lookup_embedding = not_in_lookup_embedding


    def _strip_user_profile(self, user_profiles, embedder): #Modificado pra poder receber varios user profiles de uma vez
        users = []
        for user_profile in user_profiles:
            if 'done' in user_profile and user_profile['done'] !=  'OK':
                description = ''
                user_profile = models.User(int(user_profile['user_id']))
            else:
                description = user_profile['description']
                user_profile = models.User(user_profile['id'])
            user_profile.description = description

            user = {}
            user['id'] = user_profile.id
            graphsage_embedding = self._users_embeddings_lookup.get(str(user['id']), None)
            if graphsage_embedding is None:
                graphsage_embedding = self._not_in_lookup_embedding.tolist()

            users.append(user)

        embeddings = embedder.embed(user_profiles)

        index = 0
        for user in users:
          user["embedding"] = embeddings[index].tolist() + graphsage_embedding
          index += 1

        return users


    def run(self): #Modificado pra poder usar varios user profiles de uma vez
        # Create output dir
        logging.info("Will output user embeddings to {}".format(self._user_embeddings_path))
        os.makedirs(self._user_embeddings_path, exist_ok=True)

        bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
        embedder = embeddings.UserEmbedder(bertweet_model=bertweet_model)

        length = len(list(os.scandir(self._user_profiles_path)))

        batch_size = 2  # Aqui defini o tamanho da leva como 5, mas pode ser mudado

        batch_embeddings = []
        user_profiles = []

        for fentry in tqdm(os.scandir(self._user_profiles_path), total=length):
            if len(user_profiles) < batch_size-1:
                if fentry.path.endswith(".json") and fentry.is_file():
                    with open(fentry.path) as json_file:
                        user_profiles.append(json.load(json_file))

            else:
                if fentry.path.endswith(".json") and fentry.is_file():
                    with open(fentry.path) as json_file:
                        user_profiles.append(json.load(json_file))

                batch_embeddings = self._strip_user_profile(user_profiles, embedder)
              
                for user in batch_embeddings:
                    outfile = "{}/{}.json".format(self._user_embeddings_path, user['id'])
                    with open(outfile, "w") as out_json_file:
                        logging.debug("Writing user embeddings to file {}".format(outfile))
                        json.dump(user, out_json_file)
                batch_embeddings = []
                user_profiles = []


def run(args):

    logging.info("Loading dataset") 

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    user_embeddings_path = "{}/user_embeddings".format(args.dataset_root)

    logging.info("Loading users embeddings graphsage lookup")
    with open(os.path.join(args.dataset_root, "users_graphsage_embeddings_lookup.json")) as f:
        users_embeddings_lookup = json.load(f)

    dataset = UserProfiles(
        user_profiles_path=user_profiles_path,
        user_embeddings_path=user_embeddings_path,
        embeddings_file=args.embeddings_file,
        users_embeddings_lookup=users_embeddings_lookup,
        not_in_lookup_embedding=np.zeros(len(list(users_embeddings_lookup.values())[0]))
    )

    dataset.run()


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python compute_user_embeddings.py"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str, 
        required=True
    )
    parser.add_argument(
        "--dataset-root",
        help="Output directory to export",
        dest="dataset_root",
        type=str,
        required=True
    )
    parser.add_argument(
        "--embeddings-file",
        help="Embeddings filepath",
        dest="embeddings_file",
        type=str,
        required=True
    )    
    args = parser.parse_args()
    run(args)
