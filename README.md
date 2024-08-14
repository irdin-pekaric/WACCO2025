# WSDM2025
This is a repository with supporting materials and code for our paper at WSDM2025.

The repository includes the following items:
- Literature Review:
    - The detailed classified related work list (see folder literature review).
    - The search query from the literature review (literature-review-search-query.txt).
      
- The tags of the darknet websites we excluded when constructing the darknet websites dataset (darknet-websites-excluded-tags.txt).

- The following code:
    - expand_different_capture_method.py: This code takes the HTML files from the darknet websites, extracts the text, and stores it in a CSV.
    - preprocessing_script.py: This code contains the preprocessing steps.
    - english_detection.py: This code detects text that is not in English and drops it.
    - dictionary_matching.py: This code applies dictionary matching to the raw (unprocessed) data. We applied it to the raw content because, in the preprocessing step, we remove numbers for example. Since the numbers are removed, the technical dictionary would not be able to match on hashes. However, we need the preprocessed text to feed it into the topic model.
    - comp_words_relevant_vs_not.ipynb: The code of the analysis from the most frequent words. The code identifies the most frequent words among the data items (from the ''train             dataset'') that are matched vs. not matched by our CTI dictionary. We report these for transparency, i.e. showcasing that we have objectively fine-tuned our dictionary.
    - topic_modeling.py: This code executes the topic modeling. It takes the 'preprocessed_content' from a sample and creates different visualizations and the trained topic model is saved
    - modify_model.py: This code is a template in order to modify a topic model previously trained.
    - functions.py: This code contains multiple functions used in topic_modeling.py and modify_model.py
    - prompts.py: This code only contains prompts used for the LLAMA-2 labeling of the topics

Experiments/Results:
- The topic model parameters from our experiments (topic-model-parameter-experiments.xlsx).
- The word clouds and the respective topic labels that we manually reviewed based on the suggestion of LLAMA-2.
