import argparse
import os
import pickle
import re
import time
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./output/evaluation')
    return parser.parse_args()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def extract_namus_id(path):
    if "skip" in path.lower() or "duplicate" in path.lower():
        return "skip"
    if "namus" in path:
        match = re.search(r'namus(\d+)', path)
        if match:
            return match.group(1)

    match = re.search(r'\/(\d+)-', path)
    if match:
        return match.group(1)

    match = re.search(r'_(\d+)[_\.]', path)
    if match:
        return match.group(1)

    return "unknown"

def load_or_compute_encodings(folder_path, encodings_file, model):
    if os.path.exists( 
        encodings_file
    ):   # to load cached embeddings to save time if they already exist
        print(f"Using cache")
        with open(encodings_file, 'rb') as f:
            result = pickle.load(f)

        return result
    print(f"Calculating embeddings for {folder_path}")
    embeddings = compute_embeddings(model, folder_path)

    if embeddings:
        if not os.path.exists(os.path.dirname(encodings_file)):
            os.makedirs(os.path.dirname(encodings_file))
        with open(encodings_file, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

def compute_embeddings(model, folder_path):
    embeddings = {}
    image_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if (
                file.endswith('.png')
                or file.endswith('.jpg')
                or file.endswith('.jpeg')
                or file.endswith('.bmp')
            ):
                full_path = os.path.join(root, file)
                image_files.append(full_path)

    total_files = len(image_files)

    i, j = 0, 0
    for img_path in image_files:
        i += 1
        if i % 10 == 0:
            print(f"Processing {i}/{total_files}")
        img = cv2.imread(img_path)
        if img is None:
            pil_img = Image.open(img_path)
            img_rgb = pil_img.convert('RGB')
            img = np.array(img_rgb)
            img = img[:, :, ::-1].copy()
        faces = model.get(img)
        if faces and len(faces) > 0:
            embeddings[img_path] = faces[0].embedding
            j += 1
        else:
            print(f"No face in {img_path}")
        print(f"{i} Completed, faces found in {j} images")
    return embeddings

def match_faces(query_embeddings, reference_embeddings, threshold=0.3, top_n=5):
    matches = {}

    for query_path in query_embeddings:
        query_emb = query_embeddings[query_path]
        similarities = []

        for ref_path in reference_embeddings:
            ref_emb = reference_embeddings[ref_path]
            if query_path != ref_path:
                sim = cosine_similarity(query_emb, ref_emb) # cosine similarity!!!
                if sim >= threshold:
                    pair = (ref_path, sim)
                    similarities.append(pair)
        similarities.sort(key=lambda pair: pair[1], reverse=True)

        if similarities:
            top_matches = [similarities[i] for i in range(min(top_n, len(similarities)))]
            matches[query_path] = top_matches
    return matches

def display_matches(matches, limit=5, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results_count = 0
    for query_path in matches:
        if results_count >= limit:
            break
        for i in range(len(matches[query_path][:limit])):
            ref_path, similarity = matches[query_path][i]

        results_count += 1
        if not save_dir:
            continue

        query_img = cv2.imread(query_path)
        for i in range(min(5, len(matches[query_path]))):
            (ref_path, similarity) = matches[query_path][i]
            ref_img = cv2.imread(ref_path)
            if query_img is not None and ref_img is not None:
                result_img = np.hstack([query_img, ref_img])
                query_name = os.path.splitext(os.path.basename(query_path))[0]
                ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                output_file = os.path.join(
                    save_dir, f"match_{query_name}_{ref_name}_{similarity:.4f}.jpg"
                )
                cv2.imwrite(output_file, result_img)

def main():
    args = parse_args()
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    missing_folder = "./output/MissingPersons/files/ActualPhoto/"
    unclaimed_folder = "./output/UnclaimedPersons/files/FacialCaseId/"
    unidentified_folder = "./output/UnidentifiedPersons/files/FacialCaseId/"
    query_folder = "./output/faces/"

    missing_enc_file = "./output/encodings/missing_persons.pkl"
    unclaimed_enc_file = "./output/encodings/unclaimed_persons.pkl"
    unidentified_enc_file = "./output/encodings/unidentified_persons.pkl"
    query_enc_file = "./output/encodings/query_faces.pkl"

    if not os.path.exists(query_folder):
        os.makedirs(query_folder, exist_ok=True)
        print(f"Add face images to {query_folder}, then re-run.")
        return

    query_embeddings = load_or_compute_encodings(query_folder, query_enc_file, app)

    if not query_embeddings:
        print("Add face images to the output/faces folder.")
        return

    all_reference_embeddings = {}

    for folder, enc_file, _ in [
        (missing_folder, missing_enc_file, "missing"),
        (unclaimed_folder, unclaimed_enc_file, "unclaimed"),
        (unidentified_folder, unidentified_enc_file, "unidentified"),
    ]:
        if os.path.exists(folder):
            embeddings = load_or_compute_encodings(folder, enc_file, app)
            for key in embeddings:
                all_reference_embeddings[key] = embeddings[key]
        else:
            print(f"Folder {folder} does not exist")

    if not all_reference_embeddings:
        print("Make sure to run scraper files in 'forked-namus-scraper' folder first so images are downloaded.")
        return

    matches = match_faces(
        query_embeddings, all_reference_embeddings, threshold=args.threshold, top_n=args.top_n
    )

    print(f"Showing up to {args.top_n} matches / face")
    display_matches(matches, limit=100, save_dir=args.output_dir)

    if len(matches) == 0:
        print("Check input data and/or lower threshold")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")