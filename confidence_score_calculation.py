import cv2
import numpy as np

def calculate_confidence_scores(line,depth_grad_x, depth_height, depth_width, color_image, num_samples):

    # --- Confidence Score Calculation ---
    # 1. Gradient Magnitude
    # Use a small step size to sample points along the line

    x1, y1, x2, y2 = line
    dx = (x2 - x1) / num_samples
    dy = (y2 - y1) / num_samples
    gradient_sum = 0
    sample_count = 0
    for i in range(num_samples + 1):
        x_sample = int(x1 + i * dx)
        y_sample = int(y1 + i * dy)
        if 0 <= y_sample < depth_height and 0 <= x_sample < depth_width:
            gradient_sum += depth_grad_x[y_sample, x_sample]
            sample_count += 1

    avg_gradient = gradient_sum / sample_count if sample_count > 0 else 0

    # 2. Line Length
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 3. Composite Score
    # Normalize metrics to a 0-1 range and weight them
    # Normalize avg_gradient by the max possible gradient or a known upper bound
    # The maximum possible gradient can be estimated or set as a constant
    max_possible_grad = np.max(depth_grad_x) if np.max(depth_grad_x) > 0 else 1
    normalized_grad = avg_gradient / max_possible_grad

    # Normalize line length by a reasonable max length (e.g., image height)
    normalized_length = line_length / depth_height

    # Combine with weights
    # You may need to tune these weights (e.g., 0.6 for gradient, 0.4 for length)
    confidence_score = 0.4 * normalized_grad + 0.6 * normalized_length


    # For visualization, draw the line with color based on score
    #color = (0, 0, 255) if confidence_score > 0.16 else (0, 255, 0)
    
    #cv2.line(color_image, (x1, y1), (x2, y2), color, 2)

    # Display the confidence score as text
    #cv2.putText(color_image, f"{confidence_score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return ((x1, y1, x2, y2), confidence_score)