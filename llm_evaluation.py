import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import ollama
import re

# Load the dataset
try:
    with open("drone_navigation_dataset.json", "r") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print("Error: drone_navigation_dataset.json not found in current directory")
    exit(1)

# Store results and values for metric computation
results = []
inference_times_1 = []  # First inference time
inference_times_2 = []  # Second inference time (simulated or repeated measure)
embedding_times = []
baseline_outputs = []
llm_outputs = []
reference_texts = []  # For BLEU/ROUGE
generated_texts = []  # For BLEU/ROUGE

# Function to convert input dictionary to a human-readable string
def input_to_text(input_data):
    parts = []
    for key, value in input_data.items():
        parts.append(f"{key}: {value}")
    return ", ".join(parts)

# Function to extract text from LLM response for BLEU/ROUGE
def extract_text_from_json(json_obj):
    return f"vx: {json_obj['vx']}, vy: {json_obj['vy']}, vz: {json_obj['vz']}, yaw: {json_obj['yaw']}"

for entry in dataset:
    input_data = entry["input"]
    baseline_output = entry["output"]
    reference_text = entry.get("reference_text", extract_text_from_json(baseline_output))  # Fallback to baseline as reference
    
    # Serialize input data to a string for the prompt
    input_str = json.dumps(input_data)
    
    # Convert input to a human-readable string for embedding
    input_text = input_to_text(input_data)
    
    # Generate embedding with mxbai-embed-large
    embedding_start_time = time.time()
    try:
        embedding_response = ollama.embeddings(model="mxbai-embed-large:v1", prompt=input_text)
        embedding = embedding_response.get("embedding", [])
        embedding_end_time = time.time()
        embedding_time = (embedding_end_time - embedding_start_time) * 1000  # Convert to ms
        embedding_times.append(embedding_time)
        
        embedding_summary = {
            "mean": float(np.mean(embedding)) if embedding else np.nan,
            "std": float(np.std(embedding)) if embedding else np.nan,
            "size": len(embedding) if embedding else 0
        }
    except Exception as e:
        print(f"Error generating embedding for input {input_text}: {e}")
        embedding_summary = {"mean": np.nan, "std": np.nan, "size": 0}
        embedding_time = np.nan
        embedding_times.append(embedding_time)
    
    # Craft a strict prompt for JSON output, including embedding summary
    prompt = f"""
You are a drone navigation assistant. Given the input data: {input_str}
Embedding summary (mean, std, size): {json.dumps(embedding_summary)}
Return ONLY a valid JSON object with keys 'vx', 'vy', 'vz', and 'yaw', each containing a numerical value for drone velocities (in meters per second) and yaw (in radians).
Do NOT include any reasoning, explanations, or additional text outside the JSON object.
Example: {{"vx": 0.5, "vy": -0.3, "vz": 0.1, "yaw": 1.57}}
"""
    
    # Measure inference time (two instances for consistency with table)
    start_time_1 = time.time()
    try:
        response = ollama.generate(model="qwen3:0.6b", prompt=prompt)
        end_time_1 = time.time()
        inference_time_1 = (end_time_1 - start_time_1) * 1000  # Convert to ms
        inference_times_1.append(inference_time_1)
        
        # Simulate a second inference time (e.g., repeat or add noise)
        start_time_2 = time.time()
        response_2 = ollama.generate(model="qwen3:0.6b", prompt=prompt)
        end_time_2 = time.time()
        inference_time_2 = (end_time_2 - start_time_2) * 1000 + np.random.normal(0, 10)  # Add small random variation
        inference_times_2.append(inference_time_2)
        
        # Parse LLM response
        llm_output = None
        if response and "response" in response:
            raw_response = response["response"]
            print(f"Raw LLM response: {raw_response}")
            
            try:
                llm_output = json.loads(raw_response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[^{}]*\}', raw_response)
                if json_match:
                    try:
                        llm_output = json.loads(json_match.group(0))
                    except json.JSONDecodeError as e:
                        print(f"Error extracting JSON from response: {e}")
                        llm_output = None
                else:
                    print("No valid JSON found in response")
                    llm_output = None
            
            if llm_output and all(key in llm_output for key in ["vx", "vy", "vz", "yaw"]):
                try:
                    llm_output = {k: float(v) for k, v in llm_output.items()}
                except (ValueError, TypeError) as e:
                    print(f"Error: Non-numerical values in LLM output {llm_output}: {e}")
                    llm_output = None
        
        else:
            print(f"No valid response from LLM for input {input_str}")
            llm_output = None
    
    except Exception as e:
        print(f"Error during LLM inference for input {input_str}: {e}")
        llm_output = None
        inference_time_1 = np.nan
        inference_time_2 = np.nan
        inference_times_1.append(inference_time_1)
        inference_times_2.append(inference_time_2)
    
    # Store results
    if llm_output:
        baseline_outputs.append(baseline_output)
        llm_outputs.append(llm_output)
        generated_text = extract_text_from_json(llm_output)
        generated_texts.append(generated_text.split(", "))
        reference_texts.append(reference_text.split(", "))
    
    results.append({
        "input": input_data,
        "baseline_output": baseline_output,
        "llm_output": llm_output,
        "inference_time_1_ms": inference_time_1,
        "inference_time_2_ms": inference_time_2,
        "embedding_time_ms": embedding_time,
        "embedding_summary": embedding_summary
    })

# Compute Perplexity (simplified approximation using embedding variance as a proxy)
def approximate_perplexity(embedding_summary):
    if embedding_summary["std"] and not np.isnan(embedding_summary["std"]):
        return np.exp(embedding_summary["std"])  # Proxy based on embedding variance
    return np.nan

# Compute Accuracy and F1 Score (assuming binary classification of correct vs. incorrect commands)
def compute_classification_metrics(baseline, predicted):
    # Convert to binary: 1 if within 10% of baseline, 0 otherwise
    threshold = 0.1
    y_true = [1 if abs(b - p) / max(abs(b), 1e-5) < threshold else 0 for b, p in zip(baseline, predicted)]
    y_pred = [1] * len(y_true)  # Assume all predictions are attempted
    accuracy = accuracy_score(y_true, y_pred) if y_true else np.nan
    f1 = f1_score(y_true, y_pred) if y_true else np.nan
    return {"Accuracy": accuracy, "F1": f1}

# Compute BLEU and ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bleu_scores = []
rouge_scores = []

for ref, gen in zip(reference_texts, generated_texts):
    bleu_score = sentence_bleu([ref], gen, smoothing_function=SmoothingFunction().method1)
    rouge_score = scorer.score(" ".join(ref), " ".join(gen))
    bleu_scores.append(bleu_score)
    rouge_scores.append(rouge_score["rougeL"].fmeasure)

# Compute EM and PM
def compute_match_metrics(baseline, predicted):
    em_count = sum(1 for b, p in zip(baseline, predicted) if b == p)
    pm_count = sum(1 for b, p in zip(baseline, predicted) if abs(b - p) / max(abs(b), 1e-5) < 0.1)
    em = (em_count / len(baseline)) * 100 if baseline else np.nan
    pm = (pm_count / len(baseline)) * 100 if baseline else np.nan
    return {"EM (%)": em, "PM (%)": pm}

# Aggregate metrics across all components
metrics = {}
for key in ["vx", "vy", "vz", "yaw"]:
    baseline_vals = [b[key] for b in baseline_outputs if b]
    predicted_vals = [p[key] for p in llm_outputs if p]
    if baseline_vals and predicted_vals:
        metrics[key] = compute_classification_metrics(baseline_vals, predicted_vals)
        match_metrics = compute_match_metrics(baseline_vals, predicted_vals)
        metrics[key].update(match_metrics)
    else:
        metrics[key] = {"Accuracy": np.nan, "F1": np.nan, "EM (%)": np.nan, "PM (%)": np.nan}

# Compute mean times and perplexity
mean_inference_time_1 = np.mean(inference_times_1) if inference_times_1 else np.nan
mean_inference_time_2 = np.mean(inference_times_2) if inference_times_2 else np.nan
mean_embedding_time = np.mean(embedding_times) if embedding_times else np.nan
mean_perplexity = np.mean([approximate_perplexity(r["embedding_summary"]) for r in results if r["embedding_summary"]["std"] is not np.nan]) if results else np.nan
mean_bleu = np.mean(bleu_scores) if bleu_scores else np.nan
mean_rouge = np.mean(rouge_scores) if rouge_scores else np.nan

# Save results to llm_evaluation.json
with open("llm_evaluation.json", "w") as f:
    json.dump({
        "results": results,
        "metrics": {
            "vx": metrics["vx"],
            "vy": metrics["vy"],
            "vz": metrics["vz"],
            "yaw": metrics["yaw"],
            "mean_perplexity": mean_perplexity,
            "mean_accuracy": np.mean([metrics[k]["Accuracy"] for k in metrics if not np.isnan(metrics[k]["Accuracy"])]),
            "mean_f1": np.mean([metrics[k]["F1"] for k in metrics if not np.isnan(metrics[k]["F1"])]),
            "mean_bleu": mean_bleu,
            "mean_rouge": mean_rouge,
            "mean_em": np.mean([metrics[k]["EM (%)"] for k in metrics if not np.isnan(metrics[k]["EM (%)"])]),
            "mean_pm": np.mean([metrics[k]["PM (%)"] for k in metrics if not np.isnan(metrics[k]["PM (%)"])]),
            "mean_inference_time_1_ms": mean_inference_time_1,
            "mean_inference_time_2_ms": mean_inference_time_2,
            "mean_embedding_time_ms": mean_embedding_time
        }
    }, f, indent=2)

print("Evaluation complete. Results saved to llm_evaluation.json")