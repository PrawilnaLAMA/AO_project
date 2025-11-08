#!/usr/bin/env python3
import argparse
import csv
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


def iter_images(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for r, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                yield os.path.join(r, f)


def get_embedding(image_path: str, model, mtcnn, device: str):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        logging.warning(f"No face detected in: {image_path}, skipping.")
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face).squeeze(0).cpu().numpy()
    return emb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for all images in directory and save to CSV."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root directory with images (e.g. dataset root).")
    parser.add_argument("--output-csv", required=True,
                        help="Path to output CSV file.")
    parser.add_argument("--weights", default="vggface2",
                        choices=["vggface2", "casia-webface"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"{args.input_dir} is not a directory")

    model, mtcnn = load_model(args.device, args.weights)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    first_embedding = None
    rows = []

    for img_path in iter_images(args.input_dir):
        logging.info(f"Processing {img_path}")
        emb = get_embedding(img_path, model, mtcnn, args.device)
        if emb is None:
            continue

        if first_embedding is None:
            first_embedding = emb
        rows.append((img_path, emb))

    if not rows:
        logging.warning("No embeddings computed, aborting.")
        return

    dim = len(first_embedding)
    header = ["image_path"] + [f"e{i}" for i in range(dim)]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for path, emb in rows:
            writer.writerow([path] + emb.tolist())

    logging.info(f"Saved {len(rows)} embeddings to {args.output_csv}")


if __name__ == "__main__":
    main()
