#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Acuity predictor
try:
    from insightface_acuity import create_acuity_predictor
    ACUITY_AVAILABLE = True
except ImportError as e:
    print(f"Error: Unable to import Acuity predictor: {e}")
    print("Please ensure insightface_acuity.py is in the same directory")
    sys.exit(1)

def read_lfw_pairs(pairs_path):
    pairs = []
    
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    
    # Skip possible header lines (first line might be a number)
    start_idx = 0
    if lines[0].strip().isdigit():
        start_idx = 1
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) == 3:
            # Same person
            name, idx1, idx2 = parts
            img1 = f"{name}/{name}_{int(idx1):04d}.jpg"
            img2 = f"{name}/{name}_{int(idx2):04d}.jpg"
            pairs.append((img1, img2, True))
        elif len(parts) == 4:
            # Different persons
            name1, idx1, name2, idx2 = parts
            img1 = f"{name1}/{name1}_{int(idx1):04d}.jpg"
            img2 = f"{name2}/{name2}_{int(idx2):04d}.jpg"
            pairs.append((img1, img2, False))
    
    return pairs

def extract_features_for_pairs(predictor, data_dir, pairs, score_threshold=0.2):
    all_images = set()
    for img1, img2, _ in pairs:
        all_images.add(img1)
        all_images.add(img2)
    
    print(f"Need to process {len(all_images)} unique images")
    
    features = {}
    failed_images = []
    
    for img_rel_path in tqdm(all_images, desc="Extracting features"):
        img_full_path = os.path.join(data_dir, img_rel_path)
        
        # Read image with error handling
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"\n[Error-Cannot read] Image: {img_rel_path}")
            failed_images.append(img_rel_path)
            continue
        
        # Detect faces and extract features
        try:
            faces = predictor.get(img, score_threshold=score_threshold)  # Ensure threshold is not too high
        except Exception as e:
            print(f"\n[Error-Inference exception] Processing {img_rel_path}: {e}")
            failed_images.append(img_rel_path)
            continue
        
        if len(faces) == 0:
            failed_images.append(img_rel_path)
            continue
        
        # Take features of the first detected face
        embedding = np.array(faces[0]['embedding'], dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        features[img_rel_path] = embedding
    
    # Count failures
    if failed_images:
        print(f"\nWarning: {len(failed_images)} images failed to process")
        # Print first 20 failure cases for analysis
        print("Failure examples (first 20):")
        for img in failed_images[:20]:
            print(f"  - {img}")
        if len(failed_images) > 20:
            print(f"  ... and {len(failed_images) - 20} more")
    
    # Filter valid evaluation pairs
    valid_pairs = []
    for img1, img2, is_same in pairs:
        if img1 in features and img2 in features:
            valid_pairs.append((img1, img2, is_same))
    
    print(f"\nValid evaluation pairs: {len(valid_pairs)} / {len(pairs)}")
    
    return features, valid_pairs

def compute_similarities(features, valid_pairs):
    """Compute similarity scores for evaluation pairs"""
    similarities = []
    labels = []
    
    for img1, img2, is_same in tqdm(valid_pairs, desc="Computing similarity"):
        emb1 = features[img1]
        emb2 = features[img2]
        
        # Cosine similarity
        sim = np.dot(emb1, emb2)
        similarities.append(sim)
        labels.append(1 if is_same else 0)
    
    return np.array(similarities), np.array(labels)

def evaluate_verification(scores, labels):
    """Evaluate verification performance"""
    # Find optimal threshold
    thresholds = np.arange(-1.0, 1.0, 0.001)
    best_acc = 0.0
    best_thresh = 0.0
    
    for thresh in thresholds:
        pred = (scores > thresh).astype(int)
        acc = np.mean(pred == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    # Calculate AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    
    return {
        'best_accuracy': best_acc,
        'best_threshold': best_thresh,
        'auc': auc_score,
        'scores': scores.tolist(),
        'labels': labels.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description='Validate Acuity buffalo_sc model accuracy on LFW dataset')
    parser.add_argument('--data-dir', required=True, help='LFW dataset root directory')
    parser.add_argument('--pairs-file', required=True, help='LFW evaluation pairs file (pairs.txt)')
    parser.add_argument('--det-model', type=str, default='models/acuity_models/detector',
                       help='Detection model directory (default: models/acuity_models/detector)')
    parser.add_argument('--rec-model', type=str, default='models/acuity_models/recognizer',
                       help='Recognition model directory (default: models/acuity_models/recognizer)')
    parser.add_argument('--qtype', type=str, default=None,
                       help='Quantization type, such as int8, float16, corresponding to loading model_{qtype}.quantize file')
    parser.add_argument('--output', default='acuity_lfw_results.json', help='Results output file')
    parser.add_argument('--score-threshold', type=float, default=0.2,
                       help='Face detection confidence threshold (default: 0.2)')
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.isfile(args.pairs_file):
        print(f"Error: Evaluation pairs file does not exist: {args.pairs_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Acuity buffalo_sc LFW Dataset Accuracy Validation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Evaluation pairs file: {args.pairs_file}")
    print(f"Detection model: {args.det_model}")
    print(f"Recognition model: {args.rec_model}")
    print(f"Quantization type: {args.qtype if args.qtype else 'None'}")
    print("=" * 60)
    
    # Initialize Acuity predictor
    print("\n1. Initializing Acuity buffalo_sc model...")
    try:
        predictor = create_acuity_predictor(
            det_model_dir=args.det_model,
            rec_model_dir=args.rec_model,
            qtype=args.qtype
        )
    except Exception as e:
        print(f"Acuity model initialization failed: {e}")
        sys.exit(1)
    
    # Read evaluation pairs
    print("\n2. Reading evaluation pairs...")
    pairs = read_lfw_pairs(args.pairs_file)
    print(f"   Total {len(pairs)} evaluation pairs")
    
    # Count positive and negative samples
    pos_pairs = sum(1 for _, _, is_same in pairs if is_same)
    neg_pairs = sum(1 for _, _, is_same in pairs if not is_same)
    print(f"   Positive pairs (same person): {pos_pairs}")
    print(f"   Negative pairs (different persons): {neg_pairs}")
    
    # Extract features
    print("\n3. Extracting image features...")
    features, valid_pairs = extract_features_for_pairs(
        predictor, args.data_dir, pairs, args.score_threshold
    )
    
    if len(valid_pairs) == 0:
        print("Error: No valid evaluation pairs for assessment")
        sys.exit(1)
    
    # Compute similarities
    print("\n4. Computing similarity scores...")
    scores, labels = compute_similarities(features, valid_pairs)
    
    # Evaluate performance
    print("\n5. Evaluating performance...")
    results = evaluate_verification(scores, labels)
    
    # Add metadata
    results.update({
        'model': 'buffalo_sc_acuity',
        'dataset': 'LFW',
        'evaluation_date': datetime.now().isoformat(),
        'det_model': args.det_model,
        'rec_model': args.rec_model,
        'qtype': args.qtype,
        'score_threshold': args.score_threshold,
        'total_pairs': len(pairs),
        'valid_pairs': len(valid_pairs),
        'valid_positive_pairs': int(np.sum(labels == 1)),
        'valid_negative_pairs': int(np.sum(labels == 0)),
        'data_dir': args.data_dir,
        'pairs_file': args.pairs_file
    })
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total evaluation pairs: {results['total_pairs']}")
    print(f"Valid evaluation pairs: {results['valid_pairs']}")
    print(f"Positive pairs: {results['valid_positive_pairs']}")
    print(f"Negative pairs: {results['valid_negative_pairs']}")
    print(f"Optimal threshold: {results['best_threshold']:.6f}")
    print(f"Verification accuracy: {results['best_accuracy']:.6%}")
    print("=" * 60)
    
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    save_results = results.copy()
    # Remove large arrays to reduce file size
    for key in ['scores', 'labels', 'fpr', 'tpr']:
        if key in save_results:
            save_results.pop(key, None)
    
    # Convert all NumPy types
    save_results = convert_to_serializable(save_results)
    
    with open(args.output, 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
