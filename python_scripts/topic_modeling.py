import os
import argparse
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic import BERTopic
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

#TODO FOR POTENTIAL USER CHANGE PATH IN CASE functions.py is somewhere else
path_to_functions_directory = './'
if path_to_functions_directory not in sys.path:
    sys.path.append(path_to_functions_directory)

# Import custom functions from the functions.py file and the prompts from prompts.py
from functions import *
from prompts import system_prompt, example_prompt, main_prompt

# Define command-line arguments
parser = argparse.ArgumentParser(description='Run topic modeling with BERTopic.')
parser.add_argument('--input', type=str, required=True, help='Path to the CSV file.')
parser.add_argument('--hdbscan_cluster_size', type=int, default=40, help='Minimum cluster size for HDBSCAN.')
# Type str2bool is defined in functions.py
parser.add_argument('--hdbscan_min_sample_size', type=int, default=40, help='index of how conservative docs get assigned to topics, the higher the more conservative')
parser.add_argument('--llama_labeling', type=str2bool, nargs='?', const=True, default=True, help='Define if the llama labels should be generated')
parser.add_argument('--llm_cache_path', type=str, required=True, help='In this directory the llms used are saved')
parser.add_argument('--output_directory', type=str, required=True, help='In this directory the visualizations and the models will be saved')
args = parser.parse_args()

file_name = args.input
hdbscan_min_cluster_size_args = args.hdbscan_cluster_size
hdbscan_min_sample_size_args = args.hdbscan_min_sample_size
llama_labeling = args.llama_labeling
llm_cache_path = args.llm_cache_path
output_directory = args.output_directory

# Check if the directory exists
if not os.path.exists(output_directory):
    # Create the directory if it doesn't exist
    os.makedirs(output_directory)

gpu_accel = torch.cuda.is_available()

if gpu_accel:
    try:
        # Use cuml library for GPU acceleration
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP
        print("cuml imported successfully!")
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
else:
    # Use standard libraries if cuml is not available
    from hdbscan import HDBSCAN
    from umap import UMAP
    print("No GPU accelerators Imported", flush=True)

doc_list = pd.read_csv(file_name, sep=',')['preprocessed_content'].astype('str').tolist()

model_name = file_name[:-4]
base_name = os.path.basename(model_name)
# Other possible options as encoders are:'paraphrase-distilroberta-base-v2' 'sentence-transformers/all-mpnet-base-v2' 'basel/ATTACK-BERT' # 'BAAI/bge-base-en-v1.5'
encoder_used = 'basel/ATTACK-BERT'
sanitized_encoder_used = re.sub(r'[^a-zA-Z0-9_]', '_', encoder_used)

print(f"{sanitized_encoder_used} is the encoder that was used", flush=True)
embedding_model_name = base_name + "_" + sanitized_encoder_used
base_name = f'topic_model_' + base_name

umap_n_neighbors=20 # dflt 15
umap_n_components=5 # dflt 5
umap_min_dist=0.0 # dflt 0.0
umap_metric='cosine' # dflt 'cosine'

hdbscan_min_cluster_size=hdbscan_min_cluster_size_args # dflt 15
hdbscan_metric='euclidean' # dflt 'eucledian'
hdbscan_cluster_selection_method='eom' # dflt 'eom'
hdbscan_prediction_data=True # dflt True
hdbscan_min_samples=hdbscan_min_sample_size_args # dflt is the same number as the hdbscan min cluster size

vectorizer_stop_words="english" # dflt "english"

# can be 'auto' or a number which will be later on used to enforce the number of topics
bool_enforcenrtopics = True
nr_topics_to_be_enforced = 'auto'

# Default is 10, it is the number of words per topic to be extracted
# The documentation says that usually more than 10 words give unneccessary noisy words
top_n_words = 15

current_config = {
    "encoder_used": encoder_used,
    "umap_n_neighbors": umap_n_neighbors,
    "umap_n_components": umap_n_components,
    "umap_min_dist": umap_min_dist,
    "umap_metric": umap_metric,
    "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
    "hdbscan_metric": hdbscan_metric,
    "hdbscan_cluster_selection_method": hdbscan_cluster_selection_method,
    "hdbscan_prediction_data": hdbscan_prediction_data,
    "vectorizer_stop_words": vectorizer_stop_words,
    "bool_enforcenrtopics": bool_enforcenrtopics,
    "nr_topics_to_be_enforced": nr_topics_to_be_enforced,
    "top_n_words": top_n_words,
    "llama_labeling": llama_labeling,
    "hdbscan_min_samples": hdbscan_min_samples
}

model_saving_directory = os.path.join(output_directory, 'model')
# Check if the directory exists
if not os.path.exists(model_saving_directory):
    # Create the directory if it doesn't exist
    os.makedirs(model_saving_directory)

config_dir = os.path.join(model_saving_directory, 'configs')
embeddings_directory = output_directory
print("The computation has started", flush=True)

if llama_labeling:
    print("We are using llm labeling", flush=True)
    # Loading the model
    cache_directory_llm = llm_cache_path
    llm_name = 'meta-llama/Llama-2-7b-chat-hf'
    device = get_device()
    model = load_model(llm_name, device, cache_directory_llm)

    # Llama 2 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name, cache_dir = cache_directory_llm)

    # Our text generator
    generator = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )

    # The prompts can be found in prompts.py
    prompt = system_prompt + example_prompt + main_prompt

    # Text generation with Llama 2
    llama2 = TextGeneration(generator, prompt=prompt)

# KeyBERT
keybert = KeyBERTInspired()

# MMR
mmr = MaximalMarginalRelevance(diversity=0.3)

if llama_labeling:
    # All representation models
    representation_model_grouped = {
        "KeyBERT": keybert,
        "Llama2": llama2,
        "MMR": mmr,
    }
else:
    # All representation models
    representation_model_grouped = {
        "KeyBERT": keybert,
    }


# Step 1 - Define embedding model
embedding_model = SentenceTransformer(encoder_used)

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric, random_state=42)

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        prediction_data=hdbscan_prediction_data)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words=vectorizer_stop_words)

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with
# a `bertopic.representation` model
representation_model = representation_model_grouped

# All steps together
topic_model = BERTopic(
  low_memory=False,
  calculate_probabilities=True,
  verbose=True,
  top_n_words=top_n_words,
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
)

# if already computed the precomputed embedding can be loaded as follows
# computed_embeddings = np.load(embeddings_directory + embedding_model_name)
computed_embeddings = embedding_model.encode(doc_list, show_progress_bar=True)
np.save(embeddings_directory + embedding_model_name, computed_embeddings)
print(f"Saved {embedding_model_name} in {embeddings_directory}", flush=True)

base_visualization_directory = output_directory

print(f"Starting topic modeling for {base_name}", flush=True)
    
visualization_saving_directory = os.path.join(base_visualization_directory,base_name)
wordcloud_saving_directory = f'{visualization_saving_directory}topic_wordclouds/'
if not os.path.exists(visualization_saving_directory):
    os.makedirs(visualization_saving_directory)

topics, probs = topic_model.fit_transform(doc_list, computed_embeddings)
if bool_enforcenrtopics:
    topic_model.reduce_topics(doc_list, nr_topics=nr_topics_to_be_enforced)        

if llama_labeling:
    llama2_labels_raw = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama2"].values()]

    # Sanitizing labels to contain at most 20 words
    llama2_labels_sanitized = []
    for label in topic_model.get_topics(full=True)["Llama2"].values():
        # Extracting the raw label
        raw_label = label[0][0].split("\n")[0]
    
        if raw_label:
            # Splitting the label into words
            words = raw_label.split()
            # Taking the first 20 words and joining them back into a sanitized label
            sanitized_label = ' '.join(words[:20])
        else:
            # If raw_label is empty, take the next 20 words from label[0][0]
            words = label[0][0].split()
            sanitized_label = ' '.join(words[:20])
    
        llama2_labels_sanitized.append(sanitized_label)

    # Setting sanitized topic labels
    topic_model.set_topic_labels(llama2_labels_sanitized)

try:
    topic_model.save(model_saving_directory + base_name + "_pytorch", serialization="pytorch", save_ctfidf=True, save_embedding_model=encoder_used)
    print(f"Saved {base_name} in {model_saving_directory} with pytorch serialization", flush=True)
except Exception as e:
    print(f"An error occurred while saving the model as pytorch: {e}", flush=True)
    print("Trying to save the model with safetensor serialization", flush=True)
    try:
        topic_model.save(model_saving_directory + base_name + "_safetensors", serialization="safetensors", save_ctfidf=True, save_embedding_model=encoder_used)
    except Exception as e:
        print(f"An error occurred while saving the model with safetensors: {e}", flush=True)
        print("Not saving the model but saving the configs", flush=True)

save_config(current_config, config_dir + base_name + ".config")
print(f"Saved {base_name}.config in {config_dir}", flush=True)

visualization_name = base_name

visualize_topics(topic_model, visualization_saving_directory, visualization_name)
# visualize_documents(topic_model, doc_list, visualization_saving_directory, visualization_name)
visualize_barchart(topic_model, visualization_saving_directory, visualization_name)
visualize_hierarchy(topic_model, doc_list, visualization_saving_directory, visualization_name)
save_config(current_config, os.path.join(visualization_saving_directory, f"{base_name}.config"))

# Assuming topic_model is your topic model and you want to save word clouds to 'wordcloud_saving_directory'
print("Starting wordclouds")
save_topics_wc_txt(topic_model, wordcloud_saving_directory)
