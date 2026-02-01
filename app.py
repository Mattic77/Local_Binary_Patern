import cv2
import numpy as np


def compute_lbp(image):
    """
    Compute Local Binary Pattern (LBP) for an image using fast NumPy operations.
    Uses 8-neighbor pattern with radius 1.
    
    Args:
        image: Grayscale input image
    
    Returns:
        LBP feature image
    """
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    # Get the 8 neighbors (top-left, top, top-right, right, bottom-right, bottom, bottom-left, left)
    neighbors = [
        image[0:height-2, 0:width-2],   # top-left
        image[0:height-2, 1:width-1],   # top
        image[0:height-2, 2:width],     # top-right
        image[1:height-1, 2:width],     # right
        image[2:height, 2:width],       # bottom-right
        image[2:height, 1:width-1],     # bottom
        image[2:height, 0:width-2],     # bottom-left
        image[1:height-1, 0:width-2]    # left
    ]
    
    # Get center pixels
    center = image[1:height-1, 1:width-1]
    
    # Compare neighbors with center and create binary pattern
    for i, neighbor in enumerate(neighbors):
        lbp_image[1:height-1, 1:width-1] |= ((neighbor >= center).astype(np.uint8) << i)
    
    return lbp_image


def compute_mean_lbp(image):
    """
    Compute Mean Local Binary Pattern (Mean LBP) for an image.
    Compares neighbors with the mean of all neighbors instead of center pixel.
    
    Args:
        image: Grayscale input image
    
    Returns:
        Mean LBP feature image
    """
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    # Get the 8 neighbors
    neighbors = [
        image[0:height-2, 0:width-2],   # top-left
        image[0:height-2, 1:width-1],   # top
        image[0:height-2, 2:width],     # top-right
        image[1:height-1, 2:width],     # right
        image[2:height, 2:width],       # bottom-right
        image[2:height, 1:width-1],     # bottom
        image[2:height, 0:width-2],     # bottom-left
        image[1:height-1, 0:width-2]    # left
    ]
    
    # Calculate mean of all neighbors
    neighbors_stack = np.stack(neighbors, axis=-1)
    mean_value = np.mean(neighbors_stack, axis=-1)
    
    # Compare each neighbor with the mean
    for i, neighbor in enumerate(neighbors):
        lbp_image[1:height-1, 1:width-1] |= ((neighbor >= mean_value).astype(np.uint8) << i)
    
    return lbp_image


def compute_xcs_lbp(image):
    """
    Compute eXtended Center-Symmetric Local Binary Pattern (XCS-LBP) for an image.
    Compares opposite pairs of neighbors (center-symmetric).
    
    Args:
        image: Grayscale input image
    
    Returns:
        XCS-LBP feature image
    """
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    # Get the 8 neighbors
    neighbors = [
        image[0:height-2, 0:width-2],   # top-left
        image[0:height-2, 1:width-1],   # top
        image[0:height-2, 2:width],     # top-right
        image[1:height-1, 2:width],     # right
        image[2:height, 2:width],       # bottom-right
        image[2:height, 1:width-1],     # bottom
        image[2:height, 0:width-2],     # bottom-left
        image[1:height-1, 0:width-2]    # left
    ]
    
    # Compare opposite pairs (center-symmetric)
    # Pairs: (0,4), (1,5), (2,6), (3,7)
    for i in range(4):
        diff = neighbors[i].astype(np.int16) - neighbors[i+4].astype(np.int16)
        lbp_image[1:height-1, 1:width-1] |= ((diff >= 0).astype(np.uint8) << i)
    
    return lbp_image


def main():
    """Main function to capture video and apply LBP algorithms."""
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Algorithm selection
    algorithms = {
        1: ('LBP', compute_lbp),
        2: ('Mean LBP', compute_mean_lbp),
        3: ('XCS-LBP', compute_xcs_lbp)
    }
    current_algo = 1
    
    print("=" * 60)
    print("Contrôles du clavier:")
    print("  1 - LBP (Local Binary Pattern)")
    print("  2 - Mean LBP (Mean Local Binary Pattern)")
    print("  3 - XCS-LBP (eXtended Center-Symmetric LBP)")
    print("  s - Sauvegarder l'image actuelle")
    print("  q - Quitter")
    print("=" * 60)
    print(f"\nAlgorithme actuel: {algorithms[current_algo][0]}")
    
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip horizontally to fix mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply selected algorithm
        algo_name, algo_func = algorithms[current_algo]
        lbp = algo_func(gray)
        
        # Normalize LBP for better visualization
        lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
        
        # Add text overlay showing current algorithm
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Algorithm: {algo_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the images
        cv2.imshow('Original', display_frame)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Result', lbp_normalized)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('1'):
            current_algo = 1
            print(f"\nAlgorithme changé: {algorithms[current_algo][0]}")
        elif key == ord('2'):
            current_algo = 2
            print(f"\nAlgorithme changé: {algorithms[current_algo][0]}")
        elif key == ord('3'):
            current_algo = 3
            print(f"\nAlgorithme changé: {algorithms[current_algo][0]}")
        elif key == ord('s'):
            # Save current frame
            filename_prefix = algo_name.replace(' ', '_').lower()
            cv2.imwrite(f'{filename_prefix}_frame_{frame_count}.png', lbp_normalized)
            cv2.imwrite(f'original_frame_{frame_count}.png', frame)
            print(f"\nImages sauvegardées: {filename_prefix}_frame_{frame_count}.png")
            frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
