import cv2
import os
from datetime import datetime


# Function to detect faces in an image and return face count and bounding boxes
def detect_faces(image_path, scale_factor, min_neighbors, min_size):
    # Load the Haar cascade file for face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if cascade.empty():
        raise RuntimeError("Error loading Haar Cascade XML file. Ensure the file exists and the path is correct.")

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        return 0, []

    # Resize the image to stabilize detection (if images are large)
    img = cv2.resize(img, (800, 600))  # Resize to 800x600 for consistency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    # Filter false positives by aspect ratio
    valid_faces = []
    for (x, y, w, h) in faces:
        if 0.8 <= w / h <= 1.2:  # Rough face aspect ratio validation
            valid_faces.append((x, y, w, h))

    return len(valid_faces), valid_faces


# ------ Command-line prompt for accuracy parameters ------
print("Face Detection Parameter Configuration:")
print("- scaleFactor (e.g., 1.1 - 1.5): Higher is faster but less accurate. Default: 1.2.")
print("- minNeighbors (e.g., 3 - 10): Higher values reduce false positives but risk missing faces. Default: 7.")
print("- minSize (e.g., 30 - 100): Minimum face size in pixels. Increase to ignore small patterns. Default: 50.")
print("\nPress 'Enter' to use default values for parameters.\n")

try:
    # User input or defaults
    scale_factor = float(input("Enter scaleFactor (default 1.2): ") or 1.2)
    min_neighbors = int(input("Enter minNeighbors (default 7): ") or 7)
    min_size_val = int(input("Enter minSize in pixels (default 50): ") or 50)
    min_size = (min_size_val, min_size_val)

    print("\nUsing parameters:")
    print(f"- scaleFactor: {scale_factor}")
    print(f"- minNeighbors: {min_neighbors}")
    print(f"- minSize: {min_size}\n")
except ValueError:
    print("Invalid input. Reverting to default detection parameters.\n")
    scale_factor = 1.2
    min_neighbors = 7
    min_size = (50, 50)

# ---------------------------------------------------------


# Set the directory to search in
search_dir = '/Volumes/aapark/development/ComfyUI/output/2024-12-17'

# Generate a new directory based on the current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = os.path.join(search_dir, f"verified_faces_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for images with faces
count = 0

# Open the output file
output_file = '../faces_count.txt'  # Output file remains in the current working directory
with open(output_file, 'w') as f:
    print("Processing images...")

    # Get list of files in the directory
    files = [filename for filename in os.listdir(search_dir) if filename.endswith((".jpg", ".png"))]

    # Iterate through each image in the directory
    for idx, filename in enumerate(files, start=1):
        filepath = os.path.join(search_dir, filename)

        print(f"Processing {idx}/{len(files)}: {filename}")

        # Detect faces in the current image
        face_count, face_boxes = detect_faces(filepath, scale_factor, min_neighbors, min_size)

        # Command-line feedback for detected faces
        print(f" - Detected {face_count} face(s) in {filename}")

        # If more than one face is found, process the file
        if face_count > 1:
            f.write(f"{filename} has {face_count} faces\n")

            # Read the image again for modifications
            img = cv2.imread(filepath)

            # Draw rectangles around detected faces
            for (x, y, w, h) in face_boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box with thickness 2

            # Save the modified image to the output directory
            output_filepath = os.path.join(output_dir, filename)
            cv2.imwrite(output_filepath, img)

            # Increment the counter for images with more than one face
            count += 1

            print(f" - File saved with detected faces to: {output_filepath}")
        else:
            print(f" - No action taken for: {filename}")

    print("Processing complete.")

# Final status output
print(f"\nSummary:")
print(f" - Found {count} images with more than one person in them.")
if count > 0:
    print(f" - Copied and processed files with face rectangles to: {output_dir}")
print(f" - Results written to: {output_file}")
