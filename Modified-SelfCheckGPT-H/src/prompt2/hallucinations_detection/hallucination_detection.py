import openai
from openai import OpenAI
import requests
import httpx
import json
import time
import pandas as pd
from tqdm import tqdm
import ast
from scorer import recompute_hard_labels, load_jsonl_file_to_records, score_iou, score_cor, main
import numpy as np
from langdetect import detect, LangDetectException
import re
import os
import glob


# set OpenAI API and proxies
api_key = ""
MAX_CONTEXT_LENGTH = 60000
prompt_template = """
You are an AI model output evaluation expert, responsible for detecting hallucinated words in model output and assigning accurate probability scores to each hallucination.

Below is the input information:
- **Language**: {language} (e.g., en(English), ar(Arabic), es(Spanish), etc.)
- **Question**: {question}
- **Model Output**: {output}
- **Background Knowledge** (if available): {context}

### **Task**:
Your task is to:
1. **Identify hallucinated words or phrases** in the model output based on the question and background knowledge.
   - A word or phrase is considered a hallucination if it:
     - Contradicts the background knowledge.
     - Is unverifiable or fabricated.
     - Contains logical inconsistencies.
2. **Assign a probability score** to each hallucinated word or phrase according to the following criteria:
   - **Probability > 0.7**: Severe factual errors or contradictions.
   - **Probability 0.5 - 0.7**: Unverifiable or speculative content.
   - **Probability 0.3 - 0.5**: Minor inconsistencies or unverifiable details.
   - **Probability 0.1 - 0.3**: Minor inaccuracies or vague ambiguities.
   - **Do not label words with probability ≤ 0.1** (i.e., verifiable facts).

### **Additional Instructions**:
- Do **not** mark redundant or overly generic words (e.g., "the", "a", "and") as hallucinations unless they introduce factual errors.
- Pay special attention to:
  - **Numerical data** (e.g., dates, quantities, percentages).
  - **Named entities** (e.g., people, organizations, locations).
  - **Logical contradictions** (e.g., self-contradictions within the text).
- If background knowledge is absent, base your judgment solely on internal consistency.

### **Example**:
#### Input:
- **Question**: "What year did Einstein win the Nobel Prize?"
- **Model Output**: "Einstein won the Nobel Prize in Physics in 1922 for his discovery of the photoelectric effect."
- **Background Knowledge**: "Einstein won the Nobel Prize in Physics in 1921."

#### Output:
[
    {{"word": "1922", "prob": 0.9}}
]

### **Output Format**:
Return the result as a JSON array:
[
    {{"word": <example_word>, "prob": <probability>}},
    {{"word": <another_word>, "prob": <probability>}}
]

### Important:
- Provide precise word-level annotations.
- Do not include any text or explanations outside the JSON array.
"""

def evaluate_with_selfcheck(question, output, context="", language="en", n=5, retries=3):

    if context is None:
        context = ""
    else:
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]

    language = language.lower()

    prompt = prompt_template.format(question=question, output=output, context=context, language=language)

    for attempt in range(retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "n": n
                }
            )

            time.sleep(20);
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content.strip('```json').strip('```').strip())
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON content: {content}. Error: {e}")
            else:
                print(f"Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                time.sleep(60)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(60)

    print("Retry limit exceeded, returning empty result")
    return []

# Locate word positions in the original text
def locate_word_positions(words_with_probs, model_output_text):
    ranges = []
    for item in words_with_probs:
        word = item["word"]
        prob = item["prob"]
        start_idx = model_output_text.find(word)
        while start_idx != -1:
            end_idx = start_idx + len(word)
            ranges.append((start_idx, end_idx, prob))
            start_idx = model_output_text.find(word, end_idx)
    return ranges

# Merge overlapping ranges
def merge_ranges(ranges):
    if not ranges:
        return []
    # Sort ranges by start position
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]
    for current in ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping
            new_end = max(last[1], current[1])
            new_prob = (last[2] + current[2]) / 2  # Average probabilities
            merged[-1] = (last[0], new_end, new_prob)
        else:
            merged.append(current)
    return merged

# Compute average probabilities with enhanced overlap weighting
def compute_average_probability_v3(merged_ranges, all_ranges):
    avg_probs = []
    for m_start, m_end, _ in merged_ranges:
        total_prob = 0
        total_overlap_weight = 0

        for r_start, r_end, prob in all_ranges:
            # Calculate overlap length
            overlap_start = max(m_start, r_start)
            overlap_end = min(m_end, r_end)
            overlap_length = max(0, overlap_end - overlap_start)

            # Add weighted contribution (consider overlap frequency)
            if overlap_length > 0:
                weight = overlap_length  # Base weight is overlap length
                total_prob += prob * weight
                total_overlap_weight += weight

        # Adjust probability by total weight (with enhancement factor)
        if total_overlap_weight > 0:
            final_prob = (total_prob / total_overlap_weight) ** 1.2  # Enhancing frequent overlaps
        else:
            final_prob = 0  # No overlap, probability is zero

        avg_probs.append(final_prob)
    return avg_probs

# Main function to process hallucination detection
def process_hallucination_detection(question, model_output_text, context, language):
    # Call GPT model to get hallucinated words and probabilities
    hallucinations = evaluate_with_selfcheck(question, model_output_text, context, language)
    # print("Hallucinations detected:", hallucinations)

    # Filter out hallucinations with probability <= 0.1
    hallucinations = [item for item in hallucinations if item["prob"] > 0.1]

    # Locate hallucination positions in the model output text
    hallucination_ranges = locate_word_positions(hallucinations, model_output_text)
    # print("Hallucination Ranges:", hallucination_ranges)

    # Merge overlapping ranges
    merged_ranges = merge_ranges(hallucination_ranges)
    # print("Merged Ranges:", merged_ranges)

    # Compute final probabilities for merged ranges
    final_probabilities = compute_average_probability_v3(merged_ranges, hallucination_ranges)

    # Prepare final output
    result = []
    for i, (start, end, _) in enumerate(merged_ranges):
        result.append({
            "start": start,
            "end": end,
            "prob": final_probabilities[i]
        })
    return result

def process_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

    with tqdm(total=len(input_files), desc="Processing Files", unit="file") as file_progress:
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            output_data = []

            with tqdm(total=len(data), desc=f"Processing {os.path.basename(file_path)}", unit="entry", leave=False) as entry_progress:
                for entry in data:
                    try:
                        question = entry.get("model_input", "")
                        model_output_text = entry.get("model_output_text", "")
                        context = entry.get("parsed_content", "")
                        language = entry.get("lang", "").lower()

                        soft_labels = process_hallucination_detection(
                            question, model_output_text, context, language
                        )
                        hard_labels = recompute_hard_labels(soft_labels)

                        output_entry = {
                            "id": entry.get("id"),
                            "lang": entry.get("lang"),
                            "model_input": entry.get("model_input"),
                            "model_output_text": entry.get("model_output_text"),
                            "model_id": entry.get("model_id"),
                            "soft_labels": soft_labels,
                            "hard_labels": hard_labels,
                            "model_output_logits": entry.get("model_output_logits"),
                            "model_output_tokens": entry.get("model_output_tokens")
                        }

                        output_data.append(output_entry)

                    except Exception as e:
                        print(f"Error processing entry {entry.get('id')}: {e}")

                    entry_progress.update(1)

            output_file = os.path.join(output_folder, os.path.basename(file_path))
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            file_progress.update(1)
            print(f"Processed and saved: {output_file}")

if __name__ == '__main__':
    input_folder = "data/hong/method_1/context/test/try/"
    output_folder = "data/hong/method_1/hallucination/test/"
    process_dataset(input_folder, output_folder)

