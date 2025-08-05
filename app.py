import os
import time
import cv2
import numpy as np

def calculate_min_dilation(image, extra_iterations=1, dilation_adjustment=-2):
    dilation_size = 27
    b, g, r, alpha = cv2.split(image)
    _, binary_alpha = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    while True:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated_alpha = cv2.dilate(binary_alpha, kernel, iterations=1)
        dilated_contours, _ = cv2.findContours(dilated_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(dilated_contours) == 1:
            break
        dilation_size += 1

    # Adjust the dilation size to reduce the merging
    adjusted_dilation_size = max(dilation_size + dilation_adjustment, 1)

    # Apply additional dilations with the adjusted size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adjusted_dilation_size, adjusted_dilation_size))
    for _ in range(extra_iterations):
        dilated_alpha = cv2.dilate(dilated_alpha, kernel, iterations=1)

    return adjusted_dilation_size, dilated_alpha

def process_image(image_path, extra_dilations=1, dilation_adjustment=-2):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("results", image_name)
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    original_output_path = os.path.join(output_dir, f"{image_name}.png")
    cv2.imwrite(original_output_path, image)

    min_dilation_size, dilated_alpha = calculate_min_dilation(image, extra_iterations=extra_dilations, dilation_adjustment=dilation_adjustment)
    dilated_alpha[dilated_alpha > 0] = 255
    contours, _ = cv2.findContours(dilated_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(dilated_alpha)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    mask_output_path = os.path.join(output_dir, f"{image_name}_processed_ex{extra_dilations}_mask.png")
    cv2.imwrite(mask_output_path, mask)
    print(f"Mask saved as {mask_output_path}")

    processed_image = cv2.merge((cv2.split(image)[0], cv2.split(image)[1], cv2.split(image)[2], mask))
    processed_output_path = os.path.join(output_dir, f"{image_name}_processed_ex{extra_dilations}.png")
    cv2.imwrite(processed_output_path, processed_image)
    print(f"Image processed with minimum dilation size {min_dilation_size} and saved as {processed_output_path}")

start = time.time()

# Process the uploaded image with reduced dilation adjustment
process_image('input_images/hnm.png', extra_dilations=1, dilation_adjustment=-22)

finish = time.time()
print("Time taken: ", (finish - start))
