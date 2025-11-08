#!/usr/bin/env python3
import argparse
import csv
import logging
import os

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import numpy as np
from numpy.linalg import norm

import directory_prep  # Twój plik z pytania


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def check_cuda_devices():
    logging.info("Checking CUDA devices:")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU only.")
        return False

    device_count = torch.cuda.device_count()
    logging.info(f"Found {device_count} CUDA device(s):")
    
    for i in range(device_count):
        device = torch.cuda.get_device_properties(i)
        logging.info(f"  Device {i}: {device.name}")
        logging.info(f"    Total memory: {device.total_memory / 1024**2:.0f} MB")
        logging.info(f"    CUDA Capability: {device.major}.{device.minor}")
    
    current_device = torch.cuda.current_device()
    logging.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
    return True


def load_model(device: str, weights: str):
    logging.info(f"Loading model with weights='{weights}' on device='{device}'")
    model = InceptionResnetV1(pretrained=weights).eval().to(device)
    return model


def process_image(image_path: str):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((160, 160), Image.LANCZOS)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_tensor
    except Exception as e:
        logging.warning(f"Error processing {image_path}: {str(e)}")
        return None

def get_embeddings_batch(image_paths: list, model, device: str):
    tensors = []
    valid_indices = []
    
    for idx, path in enumerate(image_paths):
        tensor = process_image(path)
        if tensor is not None:
            tensors.append(tensor)
            valid_indices.append(idx)
    
    if not tensors:
        return [], []
    
    try:
        # Stack all tensors into a batch
        batch = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        
        return embeddings, valid_indices
    except Exception as e:
        logging.warning(f"Error processing batch: {str(e)}")
        return [], []


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: trim dirs, create pairs, compare, save CSV."
    )
    parser.add_argument("--db-path", required=True,
                        help="Root directory with images.")
    parser.add_argument("--pairs-count", type=int, required=True,
                        help="Max number of pairs to generate.")
    parser.add_argument("--output-pairs", required=True,
                        help="Path to text file with generated pairs.")
    parser.add_argument("--output-csv", required=True,
                        help="Path to CSV with comparison results.")
    parser.add_argument("--weights", default="vggface2",
                        choices=["vggface2", "casia-webface"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Cosine threshold for same/different.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images to process in one batch.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    # Sprawdź dostępność CUDA przed rozpoczęciem
    if args.device == "cuda":
        has_cuda = check_cuda_devices()
        if not has_cuda:
            logging.warning("Forcing CPU usage due to CUDA unavailability")
            args.device = "cpu"

    if not os.path.isdir(args.db_path):
        raise NotADirectoryError(f"{args.db_path} is not a directory")

    # 1. Przytnij katalog (używa IMAGES_PER_FOLDER z directory_prep.py)
    logging.info("Trimming directories...")
    directory_prep.trim_paths(args.db_path)

    # 2. Generuj pary
    logging.info("Generating image pairs...")
    pairs = directory_prep.get_image_pairs(args.pairs_count, args.db_path)
    directory_prep.write_pairs_to_file(pairs, args.output_pairs)
    logging.info(f"Generated {len(pairs)} pairs and saved to {args.output_pairs}")

    # 3. Załaduj model
    model = load_model(args.device, args.weights)

    # 4. Porównaj wszystkie pary i zapisz do CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label_same", "cosine_similarity", "is_same_pred"])

        # Przygotuj listy obrazów do przetworzenia w batchu
        batch_size = args.batch_size
        total_pairs = len(pairs)
        
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            # Zbierz wszystkie obrazy z par w tym batchu
            batch_images = []
            pair_indices = []  # mapowanie indeksów batch -> indeksy par
            
            for idx, (img1, img2, _) in enumerate(batch_pairs):
                batch_images.extend([img1, img2])
                pair_indices.append((2*idx, 2*idx + 1))
            
            logging.info(f"Processing batch {batch_start//batch_size + 1}, images {batch_start*2+1}-{batch_end*2}")
            
            # Przetwórz cały batch
            embeddings, valid_indices = get_embeddings_batch(batch_images, model, args.device)
            
            if len(valid_indices) == 0:
                logging.warning(f"Skipping batch {batch_start//batch_size + 1} due to processing errors")
                continue
            
            # Porównaj pary w batchu
            for pair_idx, (img1, img2, label_str) in enumerate(batch_pairs):
                idx1, idx2 = pair_indices[pair_idx]
                
                # Sprawdź czy oba embeddingi są dostępne
                if idx1 not in valid_indices or idx2 not in valid_indices:
                    logging.warning(f"Skipping pair due to missing embedding: {img1}, {img2}")
                    continue
                
                # Znajdź właściwe embeddingi z batch results
                emb1 = embeddings[valid_indices.index(idx1)]
                emb2 = embeddings[valid_indices.index(idx2)]
                
                cos_sim = cosine_similarity(emb1, emb2)
                pred_same = cos_sim >= args.threshold
                writer.writerow([img1, img2, label_str, cos_sim, pred_same])

    logging.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()


