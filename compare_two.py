#!/usr/bin/env python3
import argparse
import logging
import os

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from numpy.linalg import norm


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


def get_embedding(img_path: str, model, mtcnn, device: str) -> np.ndarray:
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        raise RuntimeError(f"No face detected in image: {img_path}")

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face).squeeze(0).cpu().numpy()
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(norm(a - b))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two face images using embeddings."
    )
    parser.add_argument("--image1", required=True, help="Path to first image.")
    parser.add_argument("--image2", required=True, help="Path to second image.")
    parser.add_argument("--weights", default="vggface2",
                        choices=["vggface2", "casia-webface"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric", default="both",
                        choices=["cosine", "euclidean", "both"],
                        help="Which similarity metric to output.")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Decision threshold (cosine: higher ~ more similar).")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    model, mtcnn = load_model(args.device, args.weights)

    emb1 = get_embedding(args.image1, model, mtcnn, args.device)
    emb2 = get_embedding(args.image2, model, mtcnn, args.device)

    cos_sim = cosine_similarity(emb1, emb2)
    euc_dist = euclidean_distance(emb1, emb2)

    if args.metric in ("cosine", "both"):
        print(f"Cosine similarity: {cos_sim:.4f}")
        print(f"Is same (cosine >= {args.threshold}): {cos_sim >= args.threshold}")
    if args.metric in ("euclidean", "both"):
        print(f"Euclidean distance: {euc_dist:.4f}")

    logging.info("Comparison done.")


if __name__ == "__main__":
    main()
