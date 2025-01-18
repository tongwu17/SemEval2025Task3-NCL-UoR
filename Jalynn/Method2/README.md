# **SelfRefine-H** (Refined SelfCheckGPT for Hallucination Detection)

**Project Folder Structure**

```
Jalynn/
|--data/
|  |-- val/
|      |-- extract_m2/
|      |-- exknowledge_m2/
|      |-- m2_translated_keywords
|      |__ detect_gpt_m2/
|--src/
|  |-- keyphrases_extraction.ipynb
|  |-- obtain_external_knowledge.ipynb
|  |-- scorer.py
|  |__ semeval25_t3_M2.ipynb
|--stopwords/
|  |-- stopwords-ar.txt
|  |-- stopwords-eu.txt
|  |__ stopwords-zh.txt
```

**Instructions**

* data/val/: Apply the **src** codes on the val labeled dataset.
* data/val/extract_m2/:  (.jsonl files folder) Extract the key phrases from `model_input` using Hugging Face models and some other packages. More details are as follow.
* data/val/m2_translated_keywords/:  (.jsonl files folder) Translate the key phrases into English for further Rollback mechanism in obtaining external knowledge.
* data/val/exknowledge_m2/: (.jsonl files folder) Wikipedia API is used to external knowledge as context.
* data/val/detect_gpt_m2/: (.jsonl files folder) The hallucination detected results.
* src/keyphrases_extraction.ipynb: Use rules as follow to extract the key phrases
* src/obtain_external_knowledge.ipynb: Translate the key phrases into English, and then retrieve the external knowledge by Wikipedia API. If the original language cannot be searched, then use the translated one.
* src/semeval25_t3_M2.ipynb: Detect hallucinations (and evaluate the val labeled dataset).
* src/scorer.py: Evaluation code provided by organizer including compute the `hard_labels`.
* results/evaluation_results.txt: Evaluation results between detected results and val labeled datatset, using scorer.py.2;



# Overall Idea

1. Extractor Keywords
      1. Remove Stop Words:
         1. Chinese (zh): jieba
         2. Arabic (ar): Hugging Face (asafaya/bert-base-arabic)
         3. Hindi (hi): indic-nlp-library
         4. Basque (eu): tokenize - xx_ent_wiki_sm, Snowball - stopwords-eu.txt
         5. Czech (cs): stopwordsios
         6. Farsi (fa): Hazm
         7. Other Languages: spaCy model
      2. Recognize NER Entities:
         1. Hugging Face Models:
            1. Arabic (ar): asafaya/bert-base-arabic
            2. Catalan (ca): projecte-aina/roberta-base-ca-v2-cased-ner
            3. Farsi (fa): HooshvareLab/bert-fa-base-uncased-ner-arman
            4. Other Languages: FacebookAI/xlm-roberta-large-finetuned-conll03-english
         3. For Unrecognized Content, Perform Tokenization (Extract Key Nouns if Possible):
            1. Chinese (zh): jieba (tfidf-keywords)
            2. Hindi (hi): indic_tokenize
            3. Arabic (ar): Hugging Face (asafaya/bert-base-arabic)
            4. Czech (cs): Stanza
            5. Farsi (fa): Stanza
            6. Other Languages: spaCy tokenize
2. Acquire External Knowledge:
      1. Use Baidu Translate API to translate all extracted key phrases into English as a fallback mechanism for retrieval.
      2. Retrieval Rollback Mechanism:
          First, use the key phrases in the target language to search via the Wikipedia API.
        1. If the search fails, use the translated English phrases for retrieval.
      
          Note: During retrieval, there might be errors due to Traditional Chinese redirects. These need to be cleared, and results in Traditional Chinese should be forcefully converted.

      3. Extract the first 200 characters from the search results.
      
3. Use (model_input, model_output_text, context) to detect hallucination words and their probabilities via GPT-3.5.

4. Merge overlapping words and compute their probabilities using Exponentiation.

5. Create Soft Labels: Identify hallucination word positions in the model_output_text and combine them with their computed probabilities.