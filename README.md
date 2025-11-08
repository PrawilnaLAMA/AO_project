# Face Embedding & Verification Project

This project demonstrates how to:

- generate face embeddings,
- compare faces using embeddings,
- evaluate model performance on image pairs,
- log results to CSV for further analysis,

using **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** and the **InceptionResnetV1** model (pretrained on **VGGFace2** or **CASIA-WebFace**).

All scripts are parameterized and can be run directly from the command line.

---

## Project Structure

```text
AO_project/
│
├─ directory_prep.py
├─ embedding_one.py
├─ compare_two.py
├─ embeddings_to_csv.py
├─ pipeline.py
├─ logs_example.py
├─ requirements.txt
├─ .gitignore
└─ data/                 # your dataset (not included)
```

---

## Scripts Overview

### `directory_prep.py`

**Purpose**

- Prepare the dataset directory structure.
- Filter and limit images per person.
- Generate labeled image pairs for evaluation.

**Key Functions**

- `trim_paths(trim_path)`
  - Normalizes the dataset:
    - Only keeps leaf directories (no subfolders).
    - Removes directories with fewer than `IMAGES_PER_FOLDER` images.
    - In remaining directories, keeps only the first `IMAGES_PER_FOLDER` images.
- `get_image_pairs(pair_count, db_path)`
  - Creates up to `pair_count` image pairs.
  - Returns a list of tuples: `(img1_path, img2_path, is_same_person_str)`
- `write_pairs_to_file(image_pairs, output_file)`
  - Saves pairs to a text file, one pair per line.
- `is_same_person(person1, person2)`
  - Compares folder paths to decide if two images belong to the same person.

**CLI Usage**

```bash
python directory_prep.py <images_root_dir> <pairs_count> [output_file]
```

---

### `embedding_one.py`

**Purpose**

Compute a **single** face embedding and optionally save it.

**What it does**

- Loads `InceptionResnetV1(pretrained='vggface2' or 'casia-webface')`.
- Uses `MTCNN` to detect & align the face.
- Outputs:
  - Embedding vector (printed),
  - or saves it as `.npy`.

**CLI Usage**

```bash
python embedding_one.py \
  --image path/to/image.jpg \
  --output outputs/image_embedding.npy \
  --weights vggface2 \
  --device cpu \
  --log-level INFO
```

**Main Arguments**

- `--image` – input image path (required)
- `--output` – where to save `.npy` embedding (optional)
- `--weights` – `vggface2` (recommended) or `casia-webface`
- `--device` – `cpu` or `cuda`
- `--log-level` – `DEBUG` / `INFO` / `WARNING` / `ERROR`

---

### `compare_two.py`

**Purpose**

Compare **two images** to check if they are likely the same person.

**What it does**

- Loads model + MTCNN.
- Computes embeddings for both images.
- Calculates:
  - cosine similarity,
  - Euclidean distance.
- Optionally applies a **decision threshold**.

**CLI Usage**

```bash
python compare_two.py \
  --image1 path/to/img1.jpg \
  --image2 path/to/img2.jpg \
  --weights vggface2 \
  --device cpu \
  --metric both \
  --threshold 0.8 \
  --log-level INFO
```

**Output Example**

- `Cosine similarity: 0.9234`
- `Is same (cosine >= 0.8): True`
- `Euclidean distance: 0.9421`

---

### `embeddings_to_csv.py`

**Purpose**

Generate embeddings for **all images in a directory** and save them in a CSV file.

**What it does**

- Walks through `--input-dir`.
- For each image:
  - detects a face,
  - computes embedding,
  - appends a row to CSV.
- CSV columns:
  - `image_path`, `e0`, `e1`, ..., `e511` (or whatever embedding dimension is).

**CLI Usage**

```bash
python embeddings_to_csv.py \
  --input-dir data \
  --output-csv outputs/embeddings.csv \
  --weights vggface2 \
  --device cpu \
  --log-level INFO
```

---

### `pipeline.py`

**Purpose**

End-to-end script combining:

1. dataset trimming,
2. pair generation,
3. embedding computation,
4. pairwise comparison,
5. saving results to CSV.

**What it does**

- Uses `directory_prep.trim_paths` to clean up the dataset.
- Uses `directory_prep.get_image_pairs` to generate labeled pairs.
- Saves pairs to `--output-pairs`.
- For each pair:
  - computes embeddings,
  - computes cosine similarity,
  - predicts `same/different` based on `--threshold`,
  - logs everything into `--output-csv`.

**CLI Usage**

```bash
python pipeline.py \
  --db-path data \
  --pairs-count 100 \
  --output-pairs outputs/pairs.txt \
  --output-csv outputs/results.csv \
  --weights vggface2 \
  --device cpu \
  --threshold 0.8 \
  --log-level INFO
```

**CSV Output Columns**

- `img1` – path to first image
- `img2` – path to second image
- `label_same` – ground truth from directory structure
- `cosine_similarity` – computed similarity
- `is_same_pred` – predicted label using the threshold

---

### `logs_example.py`

**Purpose**

Minimal example of how logging is configured.

**What it does**

- Sets up the same logging format used in other scripts.
- Prints example log messages of different levels.
- Demonstrates progress-style logging.

**CLI Usage**

```bash
python logs_example.py --log-level DEBUG
```

Use this as a reference for consistent logging across all project scripts.

---

## Installation

From the project root:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure PyTorch is installed correctly for your environment (CPU/GPU).  
If needed, install from the official PyTorch wheels using `python -m pip` inside the virtualenv.

---

## .gitignore

This project uses a `.gitignore` configured to ignore:

- Python cache files,
- virtual environments,
- build artifacts,
- logs,
- CSV/NPY outputs,
- model/data/cache folders,
- IDE-specific files.

This keeps the repository clean and focused on source code and configuration only.
