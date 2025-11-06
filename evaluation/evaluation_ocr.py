import os
import argparse
import cv2
from tqdm import tqdm
from glob import glob
from paddleocr import PaddleOCR
import Levenshtein

from utils import load_config, update_results

def load_ground_truth(gt_txt_path):
    """Load ground truth text data from i_t.txt file."""
    ground_truth = {}
    
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    image_name = parts[0]
                    text = parts[1]
                    ground_truth[image_name] = text
    
    return ground_truth

def calculate_ned(prediction, ground_truth):
    """Calculate Normalized Edit Distance."""
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    
    edit_distance = Levenshtein.distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    if max_len == 0:
        return 0.0
    
    ned = 1.0 - (edit_distance / max_len)
    return ned



def main(config, config_name, lang):

    target_path = config.target_path
    gt_txt_path = config.gt_txt_path
    results_dir = config.results_dir
    lang_to_ppocr = {'ko': 'korean', 'jp': 'japan'}
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"results_ocr_{config_name}.json")
    
    # Load ground truth data
    ground_truth = load_ground_truth(gt_txt_path)
    
    # Initialize results structure
    results_data = {
        "accuracy": 0.0,
        "NED": 0.0,
        "results": {}
    }
    
    # Initialize OCR model
    ocr = PaddleOCR(lang=lang_to_ppocr[lang], use_gpu=True)
    
    # Process images
    image_paths = sorted(glob(os.path.join(target_path, '????.png')))
    correct_count = 0
    total_ned = 0.0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Run OCR
        result = ocr.ocr(img, det=False, rec=True, cls=False)
        
        # Extract text from OCR result
        prediction = ""
        if result and result[0]:
            # Take the text with highest confidence
            texts_with_conf = result[0]
            if texts_with_conf:
                prediction = texts_with_conf[0][0]  # Get the first text prediction
                prediction = prediction.strip()  # Clean up the text
                prediction = prediction.replace('\n', ' ')  # Replace newlines with spaces
                
                # Remove special characters from prediction
                for ch in "!\"#$%&\'()*+-./:;<=>?[\]^_`{|}~©°²½×’“”→∇∼「」​":
                    prediction = prediction.replace(ch, "")

        
        # Get ground truth for this image
        gt_text = ground_truth.get(img_name, "")
        
        # Check if prediction is correct
        is_correct = (prediction == gt_text)
        if is_correct:
            correct_count += 1
        
        # Calculate NED for this sample
        ned_score = calculate_ned(prediction, gt_text)
        total_ned += ned_score
        
        # Add result to data
        results_data["results"][img_name] = {
            "prediction": prediction,
            "ground_truth": gt_text,
            "is_correct": is_correct,
            "ned": ned_score
        }
        
        # Update accuracy and average NED
        if image_paths:  # Avoid division by zero
            results_data["accuracy"] = round(correct_count / len(image_paths), 4)
            results_data["NED"] = round(total_ned / len(image_paths), 4)
        
        # Save incrementally after each image
        update_results(results_data, output_file)
    
    # Final results update
    if image_paths:
        results_data["accuracy"] = round(correct_count / len(image_paths), 4)
        results_data["NED"] = round(total_ned / len(image_paths), 4)
    
    print(f"Evaluation complete. Results saved to {output_file}")
    print(f"Accuracy: {results_data['accuracy']:.4f}")
    print(f"NED: {results_data['NED']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate OCR performance.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True, default='ko', choices = ['ko', 'jp'])
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config_name = args.config

    main(config, config_name, args.lang)