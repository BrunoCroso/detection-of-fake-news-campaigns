#!/usr/bin/env python

import argparse
import json
import os
from typing import Dict
import models
import embeddings
import utils

import logging

import jgrapht
import pandas as pd
from tqdm import tqdm

def strip_user_profile(user_profile:Dict, embedder: embeddings.UserEmbedder) -> Dict:
    description = user_profile['description']
    user_profile = models.User(user_profile['id'])
    user_profile.description = description

    user = {}
    user['id'] = user_profile.id
    embedding = embedder.embed(user_profile)
    for i, dimension_value in enumerate(embedding):
        user['embedding_%s' % (i)] = dimension_value
    return user


def build_initial_graph():
    """Read user profiles from a json directory and create a first basic
    graph where users are vertices and edges correspond to 'follower' relationships.
    """
    logging.info("Creating graph from users")
    glove_embeddings = utils.load_glove_embeddings(args.embeddings_file)
    embedder = embeddings.UserEmbedder(glove_embeddings=glove_embeddings)

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    user_followers_path = "{}/user_followers".format(args.input_dir)

    g = jgrapht.create_graph(
        directed=True,
        allowing_self_loops=True,
        allowing_multiple_edges=True,
        any_hashable=True,
    )

    length = len(list(os.scandir(user_profiles_path)))
    for fentry in tqdm(os.scandir(user_profiles_path), total=length):
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                user_profile = json.load(json_file)
                user = strip_user_profile(user_profile, embedder)
                v = g.add_vertex(user["id"])
                g.vertex_attrs[v].update(**user)

    length = len(list(os.scandir(user_followers_path)))
    for fentry in tqdm(os.scandir(user_followers_path), total=length):
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                user_followers = json.load(json_file)
                if g.contains_vertex(user_followers["user_id"]) is False:
                    continue
                    user_profile = {"id": user_followers["user_id"], "description": ""}
                    user = strip_user_profile(user_profile, embedder)
                    v = g.add_vertex(user["id"])
                    g.vertex_attrs[v].update(**user)

                for follower in user_followers["followers"]:
                    if g.contains_vertex(follower) is False:
                        continue
                        user_profile = {"id": follower, "description": ""}
                        user = strip_user_profile(user_profile, embedder)
                        v = g.add_vertex(user["id"])
                        g.vertex_attrs[v].update(**user)

                    g.add_edge(follower, user_followers["user_id"])

    logging.info("Created graph with {} vertices".format(g.number_of_vertices))
    logging.info("Created graph with {} edges".format(g.number_of_edges))

    return g

def edges_to_df(g):
    sources = []
    targets = []

    for e in g.edges:
        u = g.edge_source(e)
        v = g.edge_target(e)
        sources.append(u)
        targets.append(v)

    return pd.DataFrame({"source": sources, "target": targets})


def vertices_to_df(g):
    data = {}

    vertices = []
    for v in g.vertices:
        vertices.append(v)

    features = list(g.vertex_attrs[v].keys())
    for f in features:
        values = []
        for v in g.vertices:
            v = g.vertex_attrs[v][f]
            values.append(v)

        data[f] = values

    return pd.DataFrame(data, index=vertices)



def run(args):
    g = build_initial_graph()
    logging.info("Create edges df")
    edges_df = edges_to_df(g)
    logging.info("Writing edges to pickle file")
    utils.write_object_to_pickle_file('edges.pkl', edges_df)
    del edges_df
    logging.info("Create vertices df")
    vertices_df = vertices_to_df(g)
    del g
    logging.info("Writing vertices to pickle file")
    utils.write_object_to_pickle_file('vertices.pkl', vertices_df)
    del vertices_df
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python users_to_graph.py --input-dir raw_data --output-file users_graph.json"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing user profiles as json files",
        dest="input_dir",
        type=str,
        default="raw_data",
    )
    parser.add_argument(
        "--output-file",
        help="Output filename to export the graph",
        dest="output_file",
        type=str,
        default="users_graph.json",
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