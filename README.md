# WSDM2025
This is a repository with supporting materials and code for our paper at WSDM2025.

The repository includes the following items:
- The tags of the darknet websites we excluded when constructing the darknet websites dataset (darknet-websites-excluded-tags.txt).
- The code of the analysis from the most frequent words. The code identifies the most frequent words among the data items (from the ''train dataset'') that are matched vs. not matched by our CTI dictionary (comp_words_relevant_vs_not.ipynb). We report these for transparency, i.e. showcasing that we have objectively fine-tuned our dictionary.
- The topic model parameters from our experiments (topic-model-parameter-experiments.xlsx).
- The code expand_different_capture_method.py which takes html files and extracts the text from it storing it into a csv.
- The code english_detection.py. The purpose of this code is, once the code was preprocessed, to detect the datapoints which are in english and to disregard the rest.
- The word clouds and the respective topic labels that we manually reviewed based on the suggestion of LLAMA-2 (to be added).
- The preprocessing_script.py code. This code contains the way we preprocess the data before putting it through the english filtering and the topic model.
- The dictionary_matching.py code. With this code we applied the dictionary matching to the raw (unprocessed) content. The reason we applied it to the raw content is that in the preprocessing step we likely got rid of most relevant technical indicators of CTI-relevant data, such as hashes.
- The code for data preprocessing, the keyword dictionary, and the code for the topic model(to be added).
- The detailed classified related work list (to be added).
- The search query from the literature review (literature-review-search-query.txt).
- General points:
    - The code are usually asking for --input and --output, the reason for this is that we are working with sensitive data so we always want to make sure to know exactly where the data is being read from and where it has to be written
    - In order for the code to work as inteded we advise to first create the raw_content, by either extracting it from html or sql files, then proceed with the preprocessing. At this point to apply the english language detector and afterwards do the dictionary matching on the raw_content. In the end the topic modeling can be done with the preprocessed data.