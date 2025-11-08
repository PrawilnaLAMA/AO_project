#!/usr/bin/env python3
import argparse
import logging
import os

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np


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
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    logging.info(f"Reading image: {image_path}")
    img = Image.open(image_path).convert("RGB")

    logging.info("Detecting and aligning face")
    face = mtcnn(img)

    if face is None:
        raise RuntimeError(f"No face detected in image: {image_path}")

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(face).squeeze(0).cpu().numpy()

    logging.info("Embedding computed")
    return emb


def save_embedding(embedding: np.ndarray, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    logging.info(f"Embedding saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute face embedding for a single image using InceptionResnetV1."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", required=False,
                        help="Path to .npy file to save embedding. If not set, only prints to stdout.")
    parser.add_argument("--weights", default="vggface2",
                        choices=["vggface2", "casia-webface"], help="Pretrained weights to use.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cpu' or 'cuda'.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    device = args.device
    model, mtcnn = load_model(device, args.weights)

    emb = get_embedding(args.image, model, mtcnn, device)

    if args.output:
        save_embedding(emb, args.output)
    else:
        # krótko wypisz pierwsze wartości
        print("Embedding shape:", emb.shape)
        print("Embedding (first 10 values):", emb[:10])


if __name__ == "__main__":
    main()
