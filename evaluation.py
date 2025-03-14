import argparse
import json
import os
import pickle
import re
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics import precision_recall_fscore_support

output_dir = "./output/evaluation"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument(
        '--validation_file', type=str, default='./output/evaluation/manual_validations.json'
    )
    parser.add_argument('--output_dir', type=str, default="./output/evaluation")
    return parser.parse_args()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_namus_id(path):
    if "skip" in path.lower() or "duplicate" in path.lower():
        return "skip"
    patterns = [ # to extract the namus id from the path, we had a lot of different formats
        r'namus(\d+)',
        r'\/(\d+)-',
        r'_(\d+)[_\.]',
        r'comparison_(\d+)_',
        r'match_(\d+)_',
    ]

    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return match.group(1)
    return "unknown"

def load_or_compute_encodings(folder_path, encodings_file, model):
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            return pickle.load(f)
    embeddings = compute_embeddings(model, folder_path)
    if embeddings:
        os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
        with open(encodings_file, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

def compute_embeddings(model, folder_path):
    embeddings = {}
    image_files = [
        os.path.join(root, file)
        for (root, _, files) in os.walk(folder_path)
        for file in files
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    for img_path in image_files:  #
        img = cv2.imread(img_path)
        if img is None:
            pil_img = Image.open(img_path)
            img = np.array(pil_img.convert('RGB'))
            img = img[:, :, ::-1].copy()
        faces = model.get(img)
        if faces and len(faces) > 0:
            embeddings[img_path] = faces[0].embedding
    return embeddings

def match_faces(embeddings_a, embeddings_b, threshold=0.5, top_n=3):
    matches = {}
    for img_a, emb_a in embeddings_a.items():
        similarities = []
        for img_b, emb_b in embeddings_b.items():
            if img_a == img_b:
                continue
            sim = cosine_similarity(emb_a, emb_b)
            if sim >= threshold:
                similarities.append((img_b, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        if similarities:
            matches[img_a] = similarities[:top_n]
    return matches

def run_compare_faces_and_get_matches(threshold=0.6, top_n=30):
    app = FaceAnalysis(name='buffalo_l') # model for face detection
    app.prepare(ctx_id=0)

    faces_folder = "./output/faces"
    missing_folder = "./output/MissingPersons/files/ActualPhoto"
    unclaimed_folder = "./output/UnclaimedPersons/files/FacialCaseId"
    unidentified_folder = "./output/UnidentifiedPersons/files/FacialCaseId"
    faces_enc_file = "./output/encodings/faces_embeddings.dat"
    missing_enc_file = "./output/encodings/missing_embeddings.dat"
    unclaimed_enc_file = "./output/encodings/unclaimed_embeddings.dat"
    unidentified_enc_file = "./output/encodings/unidentified_embeddings.dat"
    
    os.makedirs("./output/encodings", exist_ok=True)
    faces_embeddings = load_or_compute_encodings(faces_folder, faces_enc_file, app)
    all_reference_embeddings = {}

    if os.path.exists(missing_folder):
        missing_embeddings = load_or_compute_encodings(missing_folder, missing_enc_file, app)
        all_reference_embeddings.update(missing_embeddings)

    if os.path.exists(unclaimed_folder):
        unclaimed_embeddings = load_or_compute_encodings(unclaimed_folder, unclaimed_enc_file, app)
        all_reference_embeddings.update(unclaimed_embeddings)

    if os.path.exists(unidentified_folder):
        unidentified_embeddings = load_or_compute_encodings(
            unidentified_folder, unidentified_enc_file, app
        )
        all_reference_embeddings.update(unidentified_embeddings)
    matches = match_faces(faces_embeddings, all_reference_embeddings, threshold, top_n)

    return app, matches, faces_embeddings, all_reference_embeddings

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def manual_validate_matches(matches, validation_file):
    validations = {}

    if os.path.exists(validation_file):
        try:  # try/except in case the file is corrupted
            with open(validation_file, 'r') as f:
                validations = json.load(f)
        except json.JSONDecodeError:
            if os.path.exists(validation_file):
                backup_file = validation_file + ".bak"
                shutil.copy2(validation_file, backup_file)

    for query_path, match_list in matches.items():
        if query_path in validations:
            continue
        validations[query_path] = []

        for i, (ref_path, similarity) in enumerate(match_list): # this complicated function displays matches and gets user input
            query_img = cv2.imread(query_path)
            ref_img = cv2.imread(ref_path)
            target_height = 400
            (q_height, q_width) = query_img.shape[:2]
            (r_height, r_width) = ref_img.shape[:2]

            q_ratio = target_height / q_height
            r_ratio = target_height / r_height
            query_img_resized = cv2.resize(query_img, (int(q_width * q_ratio), target_height))
            ref_img_resized = cv2.resize(ref_img, (int(r_width * r_ratio), target_height))

            display_img = np.hstack([query_img_resized, ref_img_resized])
            cv2.imshow('Is this the same person? (y/n/s)', display_img)
            key = cv2.waitKey(0) & 255

            if key == ord('y'):
                validations[query_path].append((ref_path, similarity, True))
            elif key == ord('n'):
                validations[query_path].append((ref_path, similarity, False))
            elif key == ord('s'):
                validations[query_path].append((ref_path, similarity, "skip"))
            else:
                continue
            cv2.destroyAllWindows()

    with open(validation_file, 'w') as f:
        json.dump(validations, f, indent=2, cls=NumpyEncoder)
    return validations

def gather_all_similarities(matches):
    all_similarities = []
    all_labels = []

    for query_path, match_list in matches.items():
        query_id = extract_namus_id(query_path)
        for match_data in match_list:
            if len(match_data) == 2:
                ref_path, similarity = match_data
                ref_id = extract_namus_id(ref_path)

                if (
                    query_id != "unknown"
                    and ref_id != "unknown"
                    and query_id != "skip"
                    and ref_id != "skip"
                ):
                    all_similarities.append(similarity)
                    if query_id == ref_id:
                        all_labels.append("same")
                    else:
                        all_labels.append("different")
            elif len(match_data) == 3:
                ref_path, similarity, ground_truth = match_data

                if ground_truth != "skip":
                    all_similarities.append(similarity)
                    if ground_truth:
                        all_labels.append("same")
                    else:
                        all_labels.append("different")
    return all_similarities, all_labels

def calculate_performance_metrics(matches, threshold):
    all_similarities, all_labels = gather_all_similarities(matches)
    if not all_similarities:
        return {}

    y_true = [1 if label == "same" else 0 for label in all_labels]
    y_pred = [1 if sim >= threshold else 0 for sim in all_similarities]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    true_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    false_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    true_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    false_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    accuracy = (true_positives + true_negatives) / len(y_true) if len(y_true) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total_evaluations': len(all_similarities),
    }

def create_evaluation_report(metrics, output_file):
    with open(output_file, 'w') as f: # writing eval report to file
        f.write("FACE MATCHING EVALUATION SUMMARY:\n")
        f.write(f"Evaluation performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"  Precision: {metrics.get('precision', 0)*100:.1f}%\n")
        f.write(f"  Recall: {metrics.get('recall', 0)*100:.1f}%\n")
        f.write(f"  F1 Score: {metrics.get('f1', 0)*100:.1f}%\n")
        f.write(f"  Accuracy: {metrics.get('accuracy', 0)*100:.1f}%\n\n")
        f.write(f"  True Positives: {metrics.get('true_positives', 0)}\n")
        f.write(f"  False Positives: {metrics.get('false_positives', 0)}\n")
        f.write(f"  True Negatives: {metrics.get('true_negatives', 0)}\n")
        f.write(f"  False Negatives: {metrics.get('false_negatives', 0)}\n\n")
        f.write(f"Total Evaluations: {metrics.get('total_evaluations', 0)}\n")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    app, matches, _, _ = run_compare_faces_and_get_matches(threshold=0.2, top_n=15)
    if not app or not matches:
        return
    matches = manual_validate_matches(matches, args.validation_file)

    if os.path.exists(args.validation_file):
        with open(args.validation_file, 'r') as f:
            validation_data = json.load(f)
        for query_path, validations in validation_data.items():
            if query_path in matches:
                matches[query_path] = validations

    metrics = calculate_performance_metrics(matches, args.threshold)
    report_file = os.path.join(args.output_dir, 'evaluation_summary.txt')
    create_evaluation_report(metrics, report_file)

if __name__ == "__main__":
    main()