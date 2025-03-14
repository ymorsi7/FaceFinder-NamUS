import os, re, glob, face_recognition
from PIL import Image
import argparse
import requests
import time

OUTPUT_BASE = "./output"
FACES_OUTPUT = "./output/faces/namus{namus_id}-face{index}.{extension}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--input', default=None)
    parser.add_argument('--model', default='hog') # hog is faster than CNN.. would use CNN if more time
    return parser.parse_args()

def extract_namus_id(filepath):
    # looking for number in path that looks like a NamUs ID
    match = re.search(r'/(\d+)[/-]', filepath)
    if match:
        return match.group(1)
    
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    return "unknown"

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2) 

def main():
    args = parse_args()
    
    # output directory
    faces_dir = os.path.join(OUTPUT_BASE, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    
    if args.input:
        search_path = args.input
    else:
        search_path = OUTPUT_BASE
        
    image_paths = []
    
    # IMPORTANT !!! ensuring we EXCLUDE faces output directory to avoid duplicates
    for ext in ['jpg', 'jpeg', 'png']:
        found_paths = glob.glob(f"{search_path}/**/*.{ext}", recursive=True)
        filtered_paths = [p for p in found_paths if '/faces/' not in p]
        image_paths.extend(filtered_paths)
    
    print(f"Found {len(image_paths)} images")
    
    if args.limit:
        image_paths = image_paths[:args.limit]
    
    processed = 0
    faces_found = 0
    skipped = 0
    errors = 0
    
    for path in image_paths:
        try:
            namus_id = extract_namus_id(path)
            if namus_id == "unknown":
                print(f"Couldn't extract NamUs ID from {path}")
            
            image = face_recognition.load_image_file(path) # extraction
            face_locations = face_recognition.face_locations(
                image, 
                model="hog"
            )
        
            if len(face_locations) > 0:
                print(f"Found {len(face_locations)} faces")
            
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                
                height = bottom - top
                width = right - left

                padding_v = int(height * 0.2)
                padding_h = int(width * 0.2)
                
                new_top = max(0, top - padding_v) # boundaries
                new_bottom = min(image.shape[0], bottom + padding_v)
                new_left = max(0, left - padding_h)
                new_right = min(image.shape[1], right + padding_h)
                
                face_image = image[new_top:new_bottom, new_left:new_right]
                pil_image = Image.fromarray(face_image)
                
                extension = os.path.splitext(path)[1][1:].lower()
                if extension not in ['jpg', 'jpeg', 'png']:
                    extension = 'jpg'
                
                output_path = FACES_OUTPUT.format(
                    namus_id=namus_id,
                    index=i,
                    extension=extension
                )
                
                if os.path.exists(output_path) and not args.force:
                    skipped += 1
                    continue
                
                pil_image.save(output_path)
                faces_found += 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"Progress: {processed} images processed; {faces_found} faces extracted")
                
        except Exception as e:
            print(f"Couldn't process {path}: {str(e)}")
            errors += 1
    
    print("\nSummary:")
    print(f"Images processed: {processed}")
    print(f"Faces extracted: {faces_found}")
    print(f"Duplicates: {skipped}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()


# python3 scrape-data.py
# python3 scrape-files.py
# python3 process-faces.py