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

import directory_prep  # pomocnicze funkcje pracy z katalogiem

# Globalne akumulatory czasu modelu
GPU_MODEL_TIME_MS = 0.0   # suma czasów forwardów na GPU (ms)
CPU_MODEL_TIME_SEC = 0.0  # suma czasów forwardów na CPU (s)


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
    global GPU_MODEL_TIME_MS, CPU_MODEL_TIME_SEC

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

    # --- pomiar czasu samego forwardu modelu ---
    if device.startswith("cuda") and torch.cuda.is_available():
        # GPU: używamy cuda.Event do dokładnego czasu kernelów
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            emb_torch = model(batch)
        end_event.record()

        # czekamy aż GPU skończy ten batch
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # ms
        GPU_MODEL_TIME_MS += float(elapsed_ms)

        emb = emb_torch.cpu().numpy()
    else:
        # CPU: mierzymy perf_counter wokół forwardu
        start_cpu = time.perf_counter()
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        end_cpu = time.perf_counter()
        CPU_MODEL_TIME_SEC += float(end_cpu - start_cpu)

    return emb, valid_indices


def compute_embeddings_for_all_images(image_paths: list, model, device: str, batch_size: int):
    """
    Zwraca słownik {ścieżka_obrazu: embedding}.
    Obrazy są przetwarzane batchami, ale każdy obraz jest embedowany tylko raz.
    """
    embeddings_dict = {}
    total = len(image_paths)
    logging.info(f"Computing embeddings for {total} unique images...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = image_paths[start:end]

        # używamy już istniejącej funkcji batchującej (z pomiarem czasu modelu)
        batch_embeddings, valid_indices = get_embeddings_batch(batch_paths, model, device)

        if not valid_indices:
            logging.warning(f"Skipping batch {start // batch_size + 1} because all images failed")
            continue

        # valid_indices to indeksy w batch_paths, a batch_embeddings jest w tej samej kolejności
        for out_idx, img_idx in enumerate(valid_indices):
            img_path = batch_paths[img_idx]
            embeddings_dict[img_path] = batch_embeddings[out_idx]

    logging.info(f"Computed embeddings for {len(embeddings_dict)}/{total} images")
    return embeddings_dict


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
    return parser.parse_args()


def main():
    global GPU_MODEL_TIME_MS, CPU_MODEL_TIME_SEC
    args = parse_args()
    setup_logger(args.log_level)

    # zerujemy liczniki na start
    GPU_MODEL_TIME_MS = 0.0
    CPU_MODEL_TIME_SEC = 0.0

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

    embeddings_dict = compute_embeddings_for_all_images(
        all_images, model, args.device, args.batch_size
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

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label_same", "cosine_similarity", "is_same_pred"])

        skipped = 0

        for img1, img2, label_str in pairs:
            emb1 = embeddings_dict.get(img1)
            emb2 = embeddings_dict.get(img2)

            if emb1 is None or emb2 is None:
                logging.warning(
                    f"Skipping pair because embedding is missing: {img1}, {img2}"
                )
                skipped += 1
                continue

            cos_sim = cosine_similarity(emb1, emb2)
            pred_same = cos_sim >= args.threshold
            writer.writerow([img1, img2, label_str, cos_sim, pred_same])

    logging.info(
        f"Results saved to {args.output_csv}. "
        f"Skipped pairs (missing embedding): {skipped}"
    )

    # --- PODSUMOWANIE CZASÓW CPU / GPU DLA MODELU ---
    if args.device.startswith("cuda") and torch.cuda.is_available():
        logging.info(
            f"GPU model forward time (sum over batches): {GPU_MODEL_TIME_MS:.3f} ms "
            f"({GPU_MODEL_TIME_MS / 1000.0:.4f} s)"
        )
    else:
        logging.info(
            f"CPU model forward time (sum over batches): {CPU_MODEL_TIME_SEC:.4f} s"
        )


if __name__ == "__main__":
    main()
