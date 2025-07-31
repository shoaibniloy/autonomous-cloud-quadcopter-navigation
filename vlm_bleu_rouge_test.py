import cv2
import json
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from vlm import VLMClient  # Replace with the actual module name where VLMClient is defined

# Initialize VLM client (ensure the VLM server is running at this endpoint)
vlm_endpoint = "http://localhost:8080/v1/chat/completions"
instruction = "Describe the scene"  # Match the instruction used in your system
vlm_client = VLMClient(vlm_endpoint, instruction)

# Load test data
with open('references.json', 'r') as f:
    test_data = json.load(f)

generated_responses = []

# Process each test image
for item in test_data:
    image_path = item['image']
    references = item['references']
    
    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # Generate VLM response
    vlm_client.process_frame(image)
    vlm_response = vlm_client.get_response()
    
    # Store the result
    generated_responses.append({
        'image': image_path,
        'generated': vlm_response,
        'references': references
    })

# Save generated responses for inspection (optional)
with open('generated_responses.json', 'w') as f:
    json.dump(generated_responses, f, indent=4)

# Compute BLEU-4
references = [[ref.split() for ref in item['references']] for item in generated_responses]
candidates = [item['generated'].split() for item in generated_responses]
bleu_score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4 Score: {bleu_score:.4f}")

# Compute ROUGE-L
rouge = Rouge()
rouge_scores = []
for item in generated_responses:
    hyp = item['generated']
    refs = item['references']
    # Compute ROUGE-L for each reference and take the maximum
    item_scores = [rouge.get_scores(hyp, ref)[0]['rouge-l']['f'] for ref in refs]
    best_score = max(item_scores)
    rouge_scores.append(best_score)
avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
print(f"Average ROUGE-L F-Score: {avg_rouge_l:.4f}")