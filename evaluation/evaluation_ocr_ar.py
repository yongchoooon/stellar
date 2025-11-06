import os
import argparse
import cv2
from tqdm import tqdm
from glob import glob
import Levenshtein
import io
import tempfile
import imghdr
from google.cloud import vision

from utils import load_config, update_results

def set_google_credentials(credentials_file):
    """Set Google Cloud credentials environment variable."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file

def process_image_for_ocr(image_path):
    """Process image to ensure it's in JPEG format for Google Vision API."""
    img_format = imghdr.what(image_path)
    
    # Check if image is already JPEG
    if img_format and img_format.lower() in ['jpeg', 'jpg']:
        return image_path, False
    
    # Convert to JPEG if not
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.close()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, False
    cv2.imwrite(temp_file.name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return temp_file.name, True

def ocr_with_google_vision(image_path):
    """Perform OCR using Google Cloud Vision API."""
    # Process image for Vision API
    processed_path, is_temp = process_image_for_ocr(image_path)
    if not processed_path:
        return ""
    
    try:
        # Initialize Google Cloud Vision API client
        vision_client = vision.ImageAnnotatorClient()
        
        # Read image file
        with io.open(processed_path, 'rb') as image_file:
            content = image_file.read()
        
        # Create Vision API request
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            print(f"Google Vision API error: {image_path}, {response.error.message}")
            return ""
        
        # Clean up temporary file
        if is_temp and processed_path:
            try:
                os.remove(processed_path)
            except Exception as e:
                print(f"Failed to clean temporary file: {processed_path}, {e}")
        
        # Extract text from response
        if response.text_annotations:
            # Use the first annotation which contains the full detected text
            full_text = response.text_annotations[0].description
            return full_text.strip().replace('\n', ' ')
        else:
            return ""
            
    except Exception as e:
        print(f"OCR processing failed: {image_path}, {e}")
        
        # Clean up temporary file
        if is_temp and processed_path:
            try:
                os.remove(processed_path)
            except:
                pass
                
        return ""

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

def main(config, config_name):

    target_path = config.target_path
    gt_txt_path = config.gt_txt_path
    results_dir = config.results_dir
    
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
    
    # Process images
    image_paths = sorted(glob(os.path.join(target_path, '????.png')))
    correct_count = 0
    total_ned = 0.0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = os.path.basename(img_path)
        
        # Run OCR using Google Cloud Vision API
        prediction = ocr_with_google_vision(img_path)
        
        # Clean up prediction text
        prediction = prediction.strip()
        prediction = prediction.replace('\n', ' ')
        
        # Remove special characters from prediction
        for ch in "!\"#$%&\'()*+-./:;<=>?[\]^_`{|}~©°²½×'""→∇∼「」​ ":
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
    parser.add_argument('--lang', type=str, required=True, default='ar', choices = ['ar'])
    parser.add_argument('--credentials', type=str, default=None, help='Path to Google Cloud credentials JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config_name = args.config

    # Set Google Cloud credentials
    set_google_credentials(args.credentials)

    main(config, config_name)