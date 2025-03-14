import argparse
import os
import re
import shutil

def extract_namus_id(filename):
    match = re.search(r'^(\d+)[-]', filename)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    dataset_folders = [
        "./output/MissingPersons/files/ActualPhoto",
        "./output/UnidentifiedPersons/files/FacialCaseId",
        "./output/UnclaimedPersons/files/FacialCaseId",
    ]

    faces_folder = "./output/faces"  # location for extra faces
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    total_moved = 0

    for folder in dataset_folders:
        if os.path.exists(folder):
            print(f"Processing folder: {folder}") 
            groups = {}  # ID grouping (namUS gives a specific ID to each person, and naming is done as such)

            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        namusId = extract_namus_id(file)
                        if namusId:
                            fullPath = os.path.join(root, file)
                            groups.setdefault(namusId, []).append(fullPath)

            # process groups w/ 2+ images
            movedCount = 0
            for namusId, fileList in groups.items():
                if len(fileList) <= 1:
                    continue
                fileList.sort()
                for index, filePath in enumerate(fileList[1:]):
                    (_, ext) = os.path.splitext(filePath)
                    newFilename = f"namus{namusId}-face{index}{ext}"
                    newPath = os.path.join(faces_folder, newFilename)
                    if os.path.exists(newPath) and (not args.force):
                        print(f"skipping {newPath}")
                        continue
                    try:
                        shutil.copy2(filePath, newPath)
                        os.remove(filePath)
                        movedCount += 1
                    except Exception as e:
                        print(f"error moving {filePath}: {e}")

            print(f"moved {movedCount} extras from {folder} to faces folder")
            total_moved += movedCount
        else:
            print(f"folder {folder} not found")
    print(f"Total moved: {total_moved}")

if __name__ == "__main__":
    main()