import os
import argparse
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic import BERTopic
from functions import *
import itertools
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import textwrap
import re
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from scipy.cluster import hierarchy as sch
from wordcloud import WordCloud
import sys
import json
import csv
import nltk
import gensim
import gensim.corpora as corpora
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel

# Check for GPU acceleration
gpu_accel = torch.cuda.is_available()
if gpu_accel:
    # Use cuml library for GPU acceleration
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    print("If this job has a GPU the accelerators were imported correctly", flush=True)
else:
    # Use standard libraries if cuml is not available
    from hdbscan import HDBSCAN
    from umap import UMAP
    print("If this job has a GPU the accelerators were NOT imported correctly", flush=True)
    

parser = argparse.ArgumentParser(description='Modify a topic model which has been trained already.')
parser.add_argument('--input', type=str, required=True, help='Path to the directory containing the BERTopic model')
parser.add_argument('--input_config', type=str, required=True, help='Path to the .config of the topic model we are modifying')
parser.add_argument('--path_precomp_embedding', type=str, required=True, help='Path to the precomputed embeddings')
parser.add_argument('--path_data', type=str, required=True, help='Path to the data that was used to train the topic model')
parser.add_argument('--output', type=str, required=True, help='Path to the directory where the different outputs will be saved')

# Parse the arguments
args = parser.parse_args()

model_dir_path = args.input
basename = os.path.basename(model_dir_path)
output_directory = args.output

# Check if the directory exists
if not os.path.exists(output_directory):
    # Create the directory if it doesn't exist
    os.makedirs(output_directory)

# Create the path to the configs
config_path = args.input_config

(encoder_used, umap_n_neighbors, umap_n_components, umap_min_dist,
 umap_metric, hdbscan_min_cluster_size, hdbscan_metric,
 hdbscan_cluster_selection_method, hdbscan_prediction_data,
 vectorizer_stop_words, bool_enforcenrtopics, nr_topics_to_be_enforced,
 llama_labeling, hdbscan_min_samples, top_n_words) = load_config(config_path)

loaded_model = BERTopic.load(args.input, embedding_model=encoder_used)

precomp_filepath = args.path_precomp_embedding

print(f"Precomputed embeddings loaded from {precomp_filepath}", flush=True)
precomp_embeddings = np.load(precomp_filepath)

file_name = args.path_data
doc_list = pd.read_csv(file_name, sep=',')['preprocessed_content'].astype('str').tolist()
sample_df = pd.read_csv(file_name)

visualization_saving_directory = output_directory

# Modify the model
print("Modification of the model started")

hierarchical_topics = loaded_model.hierarchical_topics(doc_list)
fig = loaded_model.visualize_hierarchical_documents(doc_list, hierarchical_topics)
html_path = os.path.join(visualization_saving_directory, 'hierarchical_documents_topics.html')
fig.write_html(html_path)
print(f'{html_path} was created')

print("Modification of the model finished, proceeding to save the outputs")

