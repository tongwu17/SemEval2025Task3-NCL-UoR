**Project Folder Structure**

```
Jalynn/
|--data/
|  |-- val_new/
|  |__ detect_gpt2/
|--src/
|  |__ semeval25_t3_M2.ipynb
```

**Instructions**

* val_new/：After key phrase extraction, Wikipedia API is used to external knowledge as context.
* detect_gpt2/: the hallucination detected results



# Hallucination Detection

1. Extractor Keywords
  1. Remove Stop Words:
          1. Chinese (Zh): jieba
        2. Arabic (Ar): Hugging Face (asafaya/bert-base-arabic)
        3. Hindi (Hi): indic-nlp-library
        4. Other Languages: spaCy model
  2. Recognize NER Entities:
          1. Hugging Face Models:
            1. Arabic (Ar): asafaya/bert-base-arabic
            2. Other Languages: FacebookAI/xlm-roberta-large-finetuned-conll03-english
        2. For Unrecognized Content, Perform Tokenization (Extract Key Nouns if Possible):
            1. Chinese (Zh): jieba (tfidf-keywords)
            2. Hindi (Hi): indic_tokenize
            3. Arabic (Ar): Hugging Face (asafaya/bert-base-arabic)
            4. Other Languages: spaCy tokenize
4. Acquire External Knowledge:
      1. Use Baidu Translate API to translate all extracted key phrases into English as a fallback mechanism for retrieval.
      2. Retrieval Rollback Mechanism:
        1. First, use the key phrases in the target language to search via the Wikipedia API.
              1. If the search fails, use the translated English phrases for retrieval.
                    Note: During retrieval, there might be errors due to Traditional Chinese redirects. These need to be cleared, and results in Traditional Chinese should be forcefully converted.
      3. Extract the first 200 characters from the search results.
5. Use (model_input, model_output_text, context) to detect hallucination words and their probabilities via GPT-3.5.
6. Merge overlapping words and compute their probabilities using Exponentiation.
7. Create Soft Labels: Identify hallucination word positions in the model_output_text and combine them with their computed probabilities.