#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import time  # pomiary czasu

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import numpy as np
from numpy.linalg import norm
from torch.utils.data import Dataset, DataLoader

import directory_prep  # pomocnicze funkcje pracy z katalogiem

# (Removed per-batch timing counters: GPU_MODEL_TIME_MS / CPU_MODEL_TIME_SEC)


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def check_cuda_devices() -> bool:
    """
    Prosta diagnostyka CUDA. Zwraca True jeśli CUDA jest dostępna.
    """
    logging.info("Checking CUDA devices...")
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
    """
    Wczytuje obraz, zmienia rozmiar na 160x160 i zamienia na tensor [3,160,160] w zakresie [0,1].
    Zwraca torch.Tensor lub None (gdy wystąpi błąd).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((160, 160), Image.LANCZOS)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_tensor
    except Exception as e:
        logging.warning(f"Error processing {image_path}: {str(e)}")
        return None


def get_embeddings_batch(image_paths: list, model, device: str):
    """
    Liczy embeddingi dla listy ścieżek obrazów.
    Zwraca:
        - embeddings: np.ndarray o kształcie [N, D]
        - valid_indices: lista indeksów (w oryginalnej liście image_paths),
          dla których udało się policzyć embedding.

    Dodatkowo:
        - aktualizuje globalny GPU_MODEL_TIME_MS albo CPU_MODEL_TIME_SEC
          o czas forwardu modelu dla tego batcha.
    """
    tensors = []
    valid_indices = []

    for idx, path in enumerate(image_paths):
        tensor = process_image(path)
        if tensor is not None:
            tensors.append(tensor)
            valid_indices.append(idx)

    if not tensors:
        logging.warning("No valid images in batch.")
        return np.zeros((0, 512), dtype=np.float32), []

    batch = torch.stack(tensors).to(device)

    # Forward pass (no per-batch timing to avoid noisy/summed timings)
    with torch.no_grad():
        emb_torch = model(batch)
    # Move to CPU numpy
    emb = emb_torch.cpu().numpy()

    return emb, valid_indices


def compute_embeddings_for_all_images(image_paths: list, model, device: str, batch_size: int, num_workers: int = 4):
    """
    Zwraca słownik {ścieżka_obrazu: embedding}.
    Obrazy są przetwarzane batchami, ale każdy obraz jest embedowany tylko raz.
    """
    # Implement prefetching using DataLoader with multiple workers + pinned memory.
    class ImageDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((160, 160), Image.LANCZOS)
                arr = np.array(img)
                tensor = torch.tensor(arr).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                logging.warning(f"Failed to load image {path}: {e}")
                # Return an explicit marker that this index failed
                return path, None
            return path, tensor

    # We'll return a compact 2D numpy array (N x D) plus the corresponding list of paths
    total = len(image_paths)
    logging.info(f"Computing embeddings for {total} unique images using DataLoader (prefetch)...")

    dataloader = DataLoader(
        ImageDataset(image_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    batch_idx = 0
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    all_valid_paths = []
    emb_list = []
    gpu_embs = []

    for batch in dataloader:
        paths, tensors = batch

        # Normalize handling of tensors/list
        valid_paths = []
        valid_tensors = []
        if isinstance(tensors, torch.Tensor):
            valid_paths = list(paths)
            valid_tensors = tensors
        else:
            for p, t in zip(paths, tensors):
                if t is None:
                    logging.warning(f"Skipping broken image from dataloader: {p}")
                    continue
                valid_paths.append(p)
                valid_tensors.append(t)

        if len(valid_paths) == 0:
            logging.warning(f"Batch {batch_idx+1}: no valid images, skipping")
            batch_idx += 1
            continue

        # Ensure we have a tensor batch on CPU (pinned memory) and move to device non-blocking
        if not isinstance(valid_tensors, torch.Tensor):
            batch_cpu = torch.stack(valid_tensors).contiguous()
        else:
            batch_cpu = valid_tensors.contiguous()

        batch_gpu = batch_cpu.to(device, non_blocking=True)

        if use_cuda:
            with torch.no_grad():
                emb_torch = model(batch_gpu)
            gpu_embs.append(emb_torch.detach())
            all_valid_paths.extend(valid_paths)
        else:
            with torch.no_grad():
                emb_np = model(batch_cpu).cpu().numpy()
            # collect embeddings and paths
            emb_list.append(emb_np)
            all_valid_paths.extend(valid_paths)

        batch_idx += 1

    # After loop: consolidate into one numpy array and return
    if use_cuda:
        # ensure GPU work is finished before copying back
        torch.cuda.synchronize()

        if len(gpu_embs) > 0:
            emb_all_gpu = torch.cat(gpu_embs, dim=0)
            emb_all = emb_all_gpu.cpu().numpy()
            embeddings_array = emb_all
            paths_list = all_valid_paths
        else:
            embeddings_array = np.zeros((0, 512), dtype=np.float32)
            paths_list = []
    else:
        if len(emb_list) > 0:
            embeddings_array = np.vstack(emb_list)
            paths_list = all_valid_paths
        else:
            embeddings_array = np.zeros((0, 512), dtype=np.float32)
            paths_list = []

    logging.info(f"Computed embeddings for {len(paths_list)}/{total} images")
    return embeddings_array, paths_list


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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cpu' or 'cuda'"
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Cosine threshold for same/different.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images to process in one batch.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader worker processes to use for prefetching.")
    return parser.parse_args()


def main():
    global GPU_MODEL_TIME_MS, CPU_MODEL_TIME_SEC
    args = parse_args()
    setup_logger(args.log_level)

    # Global pipeline timer: start now to reflect entire run duration
    overall_start = time.perf_counter()

    # (Removed per-batch timing counters)

    # Sprawdzenie/sprostowanie wyboru urządzenia
    if args.device.startswith("cuda"):
        if not check_cuda_devices():
            logging.warning("Forcing CPU usage due to CUDA unavailability")
            args.device = "cpu"
    else:
        logging.info(f"Using device: {args.device}")

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

    # 4. Wylicz embeddingi dla wszystkich unikalnych obrazów
    #    i zmierz czas całego etapu (CPU + GPU + kopiowanie)
    all_images = sorted({img for img1, img2, _ in pairs for img in (img1, img2)})
    logging.info(f"Found {len(all_images)} unique images in pairs")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    emb_start = time.perf_counter()

    embeddings_array, available_images = compute_embeddings_for_all_images(
        all_images, model, args.device, args.batch_size, args.num_workers
    )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    emb_end = time.perf_counter()
    total_emb_time = emb_end - emb_start
    logging.info(
        f"Embedding generation total wall time ({args.device}): {total_emb_time:.4f} s"
    )

    # 5. Porównaj pary na podstawie wcześniej policzonych embeddingów
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # available_images is the list of image paths aligned with embeddings_array
    available_index = {p: i for i, p in enumerate(available_images)}

    # If no embeddings were produced, abort
    if embeddings_array.shape[0] == 0:
        logging.error("No embeddings available to compare. Exiting.")
        return

    # Normalize embeddings to unit vectors and compute full similarity matrix (N x N)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    emb_norm = embeddings_array / norms
    sim_matrix = emb_norm.dot(emb_norm.T)

    skipped = 0
    total_written = 0

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label_same", "cosine_similarity", "is_same_pred"])

        for img1, img2, label_str in pairs:
            idx1 = available_index.get(img1)
            idx2 = available_index.get(img2)
            if idx1 is None or idx2 is None:
                logging.warning(f"Skipping pair because embedding is missing: {img1}, {img2}")
                skipped += 1
                continue

            cos_sim = float(sim_matrix[idx1, idx2])
            pred_same = cos_sim >= args.threshold
            writer.writerow([img1, img2, label_str, cos_sim, pred_same])
            total_written += 1

    logging.info(f"Saved comparisons to {args.output_csv}. Total rows written: {total_written}. Skipped pairs: {skipped}")

    # Ensure GPU work is finished before taking final pipeline timestamp
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    overall_end = time.perf_counter()
    logging.info(f"Total pipeline wall time: {overall_end - overall_start:.4f} s")

    # (Per-batch GPU/CPU accumulated timing removed)


if __name__ == "__main__":
    main()
