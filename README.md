This is the repository of the paper _"The Dark Side of the Web: Towards Understanding Various Data Sources in Cyber Threat Intelligence"_, accepted to the WACCO 2025: the 7th Workshop on Attackers and Cybercrime Operations Co-held with IEEE European Symposium on Security and Privacy 2025 ([WACCO'25](https://www.wacco-workshop.org/)).

If you use any of our resources, you are kindly invited to cite our paper:

```
@inproceedings{schr{\"o}er2025dark,
    author = {Schr{\"o}er, Saskia Laura and Canevascini, No{\"e} and Pekaric, Irdin and Widmer, Philine and Laskov, Pavel},
    title = {{The Dark Side of the Web: Towards Understanding Various Data Sources in Cyber Threat Intelligence}},
    booktitle = {WACCO 2025: the 7th Workshop on Attackers and Cybercrime Operations Co-held with IEEE European Symposium on Security and Privacy 2025},
    year = {2025}
}

```

# Description

This is an official repository for the supplementary materials for the paper "The Dark Side of the Web: Towards Understanding Various Data Sources in Cyber Threat Intelligence". It contains the following files:

(1) Literature Review:
  - The detailed classified related work list (see folder literature review).
  - The search query from the literature review (literature-review-search-query.txt)

(2) The tags of the darknet websites we excluded when constructing the darknet websites dataset (darknet-websites-excluded-tags.txt)
    
(3) The following code:
  - Expand_different_capture_method.py: This code takes the HTML files from the darknet websites, extracts the text, and stores it in a CSV.
  - Preprocessing_script.py: This code contains the preprocessing steps.
  - english_detection.py: This code detects text that is not in English and drops it.
  - dictionary_matching.py: This code applies dictionary matching to the raw (unprocessed) data. We use the raw content because the preprocessing step removes elements like numbers, which would otherwise prevent the technical dictionary from matching on hashes. However, the preprocessed text is still needed for input into the topic model.
  - comp_words_relevant_vs_not.ipynb: The code of the analysis from the most frequent words. The code identifies the most frequent words among the data items that are matched vs. not matched by our CTI dictionary. We report these for transparency, i.e. showcasing that we have objectively fine-tuned our dictionary.
  - topic_modeling.py: This code executes the topic modeling. It takes the 'preprocessed_content' from a sample and creates different visualizations and the trained topic model is saved for further analysis and/or modification.
  - modify_model.py: This code is a template in order to modify a topic model previously trained.
  - functions.py: This code contains multiple functions used in topic_modeling.py and modify_model.py
  - prompts.py: This code only contains prompts used for the LLAMA-2 labeling of the topics
  - pip_requirements_for_wsdm.txt: This is not code but it contains the pip requirements used for the topic modeling. In the code we used GPU acceleration with cuml, which might need to be downloaded separately, for more informations see: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#speed-up-umap

(4) Experiments/Results:
  - The topic model parameters from our experiments (topic-model-parameter-experiments.xlsx).
  - The word clouds and the respective topic labels that we manually reviewed based on the suggestion of LLAMA-2 (in the folder topic_model_experiments).
    
### Contact

If you have any inquiry about our research (e.g., about the literature review, data sources, source code or experiments) feel free to contact the first author, [Saskia Laura Schröer] (saskia.schroeer@uni.li)




