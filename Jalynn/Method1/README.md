# **CLAUDE-Refiner** (RefChecker Framework Enhanced by CLAUDE)

**Project Folder Structure**

```
Jalynn/
|--data/
|  |-- val/
|      |-- exknowledge_m1/
|      |-- etract_m1/
|      |__ detect_m1/
|  |-- test/
|      |-- detect_m1/
|--src/
|  |-- RefChecker/
|  |-- external_knowledge_google.ipynb
|  |-- keyphrases_extraction_gpt.ipynb
|  |-- scorer.py
|  |__ semeval25_t3.ipynb
|--result/
|  |__ evaluation_results.txt
```

**Instructions**

* data/val/: Apply the **src** codes on the val labeled dataset.
* data/val/extract_m1/: (.jsonl files folder) Extract the key phrases from `model_input` using OpenAI API , and then for the further external knowledge.
* data/val/exknowledge_m1/：(.jsonl files folder) After key phrase extraction, Google API is used to external knowledge as context.
* data/val/detect_m1/: (.jsonl files folder) The hallucination detected results.
* data/test/detect_m1/: (.jsonl files folder) The hallucination detected results on **test unlabeled dataset**.
* src/RefChecker/: https://github.com/amazon-science/RefChecker?tab=readme-ov-file 
* src/keyphrases_extraction_gpt.ipynb: Use OpenAI API to extract key phrases.
* src/external_knowledge_google.ipynb: Retrieve external knowledge by extracted key phrases using Google Custom Search Engine (Google Custom Search JSON API).
* src/semeval25_t3.ipynb: Detect hallucinations (and evaluate the val labeled dataset).
* src/scorer.py: Evaluation code provided by organizer including compute the `hard_labels`.
* results/evaluation_results.txt: Evaluation results between detected results and val labeled datatset, using scorer.py.



# Overall Idea

**The method adopts the concept of RefChecker and is primarily divided into Extractor and Checker components. However, I am unable to directly apply the GitHub method (https://github.com/amazon-science/RefChecker/tree/main/refchecker) due to the following issues:**

1. The GitHub implementation relies on the OpenAI API, which requires a proxy for usage in China.
2. RefChecker does not achieve the goal of hallucination position detection.
3. RefChecker uses its own set of references for comparison, but the references I have are limited, some are inaccessible and just can be the context.

**Based on the above, I only use the conceptual framework of RefChecker and some of its key prompts for improvement and optimization. In my method, I primarily use the Anthropic API (Claude).**

**My idea of solving the problems:**
1. **Claim Extraction:** Claims are extracted by using the Anthropic API to process both `model_output_text` and `model_input`. Each claim is a merged representation derived from a triple-structured knowledge breakdown. (This uses prompts from RefChecker's Extractor component).

2. **Primary Reference Acquisition:** Based on the claims obtained in Step 1, I use a self-verification approach. Each claim is processed through the Anthropic API for self-verification to obtain factually consistent statements and even supplemental content. The returned statements are combined with external knowledge I retrieved earlier using the Google API. This forms a complete set of references for subsequent verification checks.

3. **Verification of Claims, Model Input, Model Output Text, and References:** The extracted claims, references from Step 2, and `model_input` are verified. The Anthropic API is employed with specific prompts to identify hallucinated words in `model_output_text` and estimate their hallucination probability.

4. **Soft Label Generation:** Based on the hallucinated words identified above, their positions in `model_output_text` are mapped. Using the probabilities obtained, a set of `soft_labels` is generated.