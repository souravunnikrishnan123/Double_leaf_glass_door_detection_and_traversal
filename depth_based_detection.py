    
        # -------------------- STEP 2: DEPTH GRADIENT + HOUGH (NEW CODE) --------------------
        '''
        # Convert to float32 and scale to meters if needed
        depth_float = depth_image.astype(np.float32)
        depth_float /= 1000.0  # Convert mm to meters if necessary


        

        valid_mask = np.isfinite(depth_float) & (depth_float > MIN_DEPTH) & (depth_float < MAX_DEPTH)
        #inpaint_mask = (~valid_mask).astype(np.uint8)
        #depth_inpainted = cv2.inpaint(depth_float, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        
        depth_valid = np.where(valid_mask, depth_float, 0)  # Or use median filling instead of 0

        depth_filtered = cv2.bilateralFilter(depth_valid, d=7, sigmaColor=50, sigmaSpace=75)

        # Get horizontal gradient
        depth_grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=5)
        depth_grad_x = np.absolute(depth_grad_x)
        depth_grad_x[~valid_mask] = 0
        #depth_grad_x = cv2.normalize(depth_grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Dynamic threshold (e.g., 20% of max)
        #thresh_val = 0.05 * np.max(depth_grad_x)
        valid_grad_vals = depth_grad_x[valid_mask]
        mean_val = np.mean(valid_grad_vals)
        std_val = np.std(valid_grad_vals)
        k = 1  # You can tune this value

        thresh_val = mean_val + k * std_val
        _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
        depth_edges = depth_edges.astype(np.uint8)
        cv2.imshow("Depth Edges", depth_edges)

                # Try to visualize what Hough sees

        # Use Hough Transform on depth-based edges
        depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=20)

        debug_copy = color_image.copy()
        if depth_lines is not None:
            for line in depth_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            print("[DEBUG] No depth-based Hough lines found.")
        cv2.imshow("Depth Hough Lines", debug_copy)

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if 70 < abs(angle) < 110:  # vertical-ish
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            d = get_z_depth(depth_frame,x_center, y_center)

            if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Red = depth-based lines
        '''