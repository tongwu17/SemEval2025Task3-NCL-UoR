# Modified-SelfCheckGPT-H

**Project Folder Structure**

```
NCL-UoR/
|--data/
|  |-- google_custom_search_api_abstract_data/
|      |-- test/
|          |-- context/
|              |__ exknowledge_m1/
|          |-- hallucinations/
|              |__ detect_gpt4_m1and2
|          |__ keywords/
|              |__ extract_m1
|      |-- val/
|          |-- context/
|              |__ exknowledge_m1/
|          |__ keywords/
|              |__ extract_m1
|  |-- google_custom_search_api_data/
|      |-- context/
|          |__ test/
|      |-- hallucinations/
|          |__ test/
|      |__ keywords/
|          |__ test/
|          |__ val/
|  |-- wikipedia_api_data/
|      |-- test/
|          |-- context/
|              |__ exknowledge_m2/
|          |-- hallucinations/
|              |-- detect_gpt_m2/
|              |-- detect_gpt4o_m2/
|              |__ test_detect_gpt4/
|          |__ keywords/
|              |__ extract_m2/
|      |-- val/
|          |-- context/
|              |-- exknowledge_m2/
|              |__ m2_translated_keywords/
|          |-- hallucinations/
|              |--detect_gpt3.5_m2/
|              |__ detect_gpt4o_m2/
|          |__ keywords/
|              |__ extract_m2/
|--result/
|  |-- eval_results.csv
|  |-- evaluation_results_gpt3.5.txt
|  |__ evaluation_results_gpt4o_custom_rules.txt
|--src/
|  |-- prompt1/
|      |-- task3-prompting-approach.ipynb
|  |-- prompt2/
|  |-- external_knowledge/
|      |-- obtain_external_knowledge.ipynb
|  |-- keyphrases_extraction/
|      |-- keyphrases_extraction.ipynb
|  |__ hallucination_detection/
|      |-- semeval25_t3_M2-GPT4o.ipynb
|      |-- semeval25_t3_M2-GPT4.ipynb
|      |-- semeval25_t3_M2.ipynb
|      |__ hallucination_detection.py
|  |-- scorer.py
|--stopwords/
|  |-- stopwords-ar.txt
|  |-- stopwords-eu.txt
|  |__ stopwords-zh.txt
```

**Instructions**

* data/googlel_custom_search_api_abstract_data/: Using **GPT-3.5** to extract keywords and then obtain external knowledge via **Google CSE API** (abstract of searched page)
* data/google_custom_search_api_data/: Using **GPT-3.5** to extract keywords and then obtain external knowledge via **Google CSE API** (content of the searched page)
* data/wikipedia_api_data/: Using **customed rules (hugging face models, spaCy and TF-IDF)** to extract keywords and then obtain external knowledge via **Wikipedia API**
* src/prompt1/: Using **YAKE** to extract keywords and then obtain external knowledge via **Wikipedia API**. And using **GPT-3.5** to detect hallucinations.
* src/prompt2/keyphrases_extraction.ipynb: Use rules as follow to extract the key phrases
* src/prompt2/external_knowledge/obtain_external_knowledge.ipynb: Translate the key phrases into English, and then retrieve the external knowledge by Wikipedia API. If the original language cannot be searched, then use the translated one.
* src/prompt2/hallucinations_detection/semeval25_t3_M2.ipynb: Detect hallucinations (and evaluate the labeled dataset) via **gpt-3.5-turbo** with **customed rules and Wikipedia API external knowledge**.
* src/prompt2/hallucinations_detection/semeval25_t3_M2-GPT4o.ipynb: Detect hallucinations (and evaluate the labeled dataset) via **gpt-4o** with **customed rules and Wikipedia API external knowledge**.
* src/prompt2/hallucinations_detection/semeval25_t3_M2-GPT4.ipynb: Detect hallucinations (and evaluate the labeled dataset) via **gpt-4** with **GPT-3.5 extraction and Google CSE API (abstract) external knowledge**.
* src/prompt2/hallucinations_detection/hallucination_detection.py:  Detect hallucinations (and evaluate the labeled dataset) via **gpt-4o** with **GPT-3.5 extraction and Google CSE API (page content) external knowledge**.
* src/scorer.py: Evaluation code provided by organizer including compute the `hard_labels`.
* results/evaluation_results_gpt3.5.txt: val set results (evaluation results of semeval25_t3_M2.ipynb)
* results/evaluation_results_gpt4o_custom_rules.txt: val set results (evaluation results of semeval25_t3_M2-GPT4o)
* results/eval_results.csv: val set results (evaluation results of prompt1/task3-prompting-approach.ipynb)



# Overall Idea

1. Extractor Keywords
2. Acquire External Knowledge
3. Detect hallucination via GPT (gpt-3.5-turbo, gpt-4o, gpt-4).
4. Merge overlapping words and compute their probabilities.
5. Create Soft Labels: Identify hallucination word/tokens positions in the model_output_text and combine them with their computed probabilities.