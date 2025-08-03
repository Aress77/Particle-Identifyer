
import cv2 as cv
import numpy as np
import os
import glob

# Edit code based on your image/preference
# --- Parameters ---
blockSize = 15
C = 2
min_area = 20
max_area = 5000
alpha = 0.1  # Contrast
beta = -100   # Brightness

use_clahe = True  # Toggle CLAHE on/off


def preprocess_image(img):
    """Normalize and enhance contrast."""
    img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_8bit)
    else:
        return cv.convertScaleAbs(img_8bit, alpha=alpha, beta=beta)

def threshold_and_clean(img):
    """Apply adaptive thresholding and morphological filtering."""
    blurred = cv.medianBlur(img, 7)

    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Morphological noise cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)

    return cleaned


def analyze_and_draw(contours, adjusted_img, name_wo_ext, output_folder):
    """Filter contours, draw them, and save result image."""
    filtered = [c for c in contours if min_area <= cv.contourArea(c) <= max_area]
    print(f"Detected {len(filtered)} particles.")

    # Draw contours
    img_with_contours = cv.cvtColor(adjusted_img.copy(), cv.COLOR_GRAY2BGR)
    cv.drawContours(img_with_contours, filtered, -1, (0, 255, 255), 1)

    # Save result image
    contour_path = os.path.join(output_folder, f"{name_wo_ext}_contours.tif")
    cv.imwrite(contour_path, img_with_contours)


def process_image(image_path, output_folder):
    """Full pipeline for a single image."""
    filename = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(filename)[0]
    print(f"\nProcessing {filename}...")

    img = cv.imread(image_path, -1)
    if img is None:
        print("Failed to load:", filename)
        return

    adjusted = preprocess_image(img)

    # Save adjusted image
    adjusted_path = os.path.join(output_folder, f"{name_wo_ext}_adjusted.tif")
    cv.imwrite(adjusted_path, adjusted)

    denoised = cv.medianBlur(adjusted, 3)  # Or use 5 if noise is worse

    # Then pass denoised to threshold_and_clean instead of adjusted
    cleaned = threshold_and_clean(denoised)

    # Find contours
    contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    analyze_and_draw(contours, adjusted, name_wo_ext, output_folder)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "path") #replace with your own path
    output_folder = os.path.join(input_folder, "processed")
    os.makedirs(output_folder, exist_ok=True)

    for image_path in glob.glob(os.path.join(input_folder, "*.tif")):
        process_image(image_path, output_folder)

    print("\n All images processed and saved.")


if __name__ == "__main__":
    main()
