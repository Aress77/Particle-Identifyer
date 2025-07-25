#!/Users/ahri/Documents/Summer OPALS/Group 18/venv/bin/python


import cv2 as cv
import numpy as np

# Load 16-bit image as-is
image1 = "E21_07_22_25_16h40m_28s_ms234__.tif" 
img = cv.imread(image1, -1)

# Show basic info
print('Data type:', img.dtype)
print('Shape:', img.shape)
print('Min pixel value:', img.min())
print('Max pixel value:', img.max())

# Normalize the image to 0–255 and convert to uint8
img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Median blur to reduce noise
median_blur_img = cv.medianBlur(img_8bit, 5)
cv.imshow('Median Blurring', median_blur_img)

# Adaptive Thresholding
blockSize = 15  # Must be odd
C = 2
thresh_img = cv.adaptiveThreshold(
    median_blur_img,
    maxValue=255,
    adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv.THRESH_BINARY_INV,
    blockSize=blockSize,
    C=C
)
cv.imshow("Adaptive Thresholding", thresh_img)
cv.imwrite('thresh_blur_image.tif', thresh_img)

# Post-threshold median blur to clean noise
post_thresh_img = cv.medianBlur(thresh_img, 5)
cv.imshow('Post Median Blurring', post_thresh_img)
cv.imwrite('post_thresh_blur_image.tif', post_thresh_img)

# Find contours
contours, _ = cv.findContours(post_thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Count particles by area
min_area = 5
max_area = 1200
cell_count = sum(min_area <= cv.contourArea(c) <= max_area for c in contours)

print(f"Number of particles: {cell_count}")
print('Particle size range:', min_area, 'to', max_area)
print('Thresholding: ADAPTIVE_THRESH_MEAN_C | blockSize =', blockSize, '| C =', C)

# Draw contours
img_with_contours = cv.cvtColor(img_8bit.copy(), cv.COLOR_GRAY2BGR)
cv.drawContours(img_with_contours, contours, -1, (0, 255, 255), 1)
cv.imshow("Image with Contours", img_with_contours)

# Wait for key or Ctrl+C
try:
    print("Press any key or Ctrl+C to exit...")
    while True:
        if cv.waitKey(100) != -1:
            break
except KeyboardInterrupt:
    print("\n[INFO] Exit with Ctrl+C")
finally:
    cv.destroyAllWindows()
