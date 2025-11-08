#!/usr/bin/env python3
import argparse
import csv
import logging
import os

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from numpy.linalg import norm

import directory_prep  # Twój plik z pytania


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def load_model(device: str, weights: str):
    logging.info(f"Loading model with weights='{weights}' on device='{device}'")
    model = InceptionResnetV1(pretrained=weights).eval().to(device)
    mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=device)
    return model, mtcnn


def get_embedding(image_path: str, model, mtcnn, device: str):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        logging.warning(f"No face detected in {image_path}, skipping.")
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face).squeeze(0).cpu().numpy()
    return emb


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
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

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
    model, mtcnn = load_model(args.device, args.weights)

    # 4. Porównaj wszystkie pary i zapisz do CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label_same", "cosine_similarity", "is_same_pred"])

        for img1, img2, label_str in pairs:
            logging.info(f"Comparing:\n  {img1}\n  {img2}")

            emb1 = get_embedding(img1, model, mtcnn, args.device)
            emb2 = get_embedding(img2, model, mtcnn, args.device)

            if emb1 is None or emb2 is None:
                logging.warning("Skipping pair due to missing embedding.")
                continue

            cos_sim = cosine_similarity(emb1, emb2)
            pred_same = cos_sim >= args.threshold
            writer.writerow([img1, img2, label_str, cos_sim, pred_same])

    logging.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()


