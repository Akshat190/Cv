import cv2
import numpy as np
import os

def create_blurred_versions():
    # Specify the exact path for input and output
    input_image_path = r"C:\Users\Admin\Desktop\cv\images\true.jpg"
    output_dir = r"C:\Users\Admin\Desktop\cv\images"
    
    # Read the input image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not read image at {input_image_path}")
        return
    
    # Create different blur levels
    # 1. Low blur (Gaussian)
    blur_low = cv2.GaussianBlur(img, (5, 5), 1.5)
    cv2.imwrite(os.path.join(output_dir, "blur_low.jpg"), blur_low)
    
    # 2. Moderate blur (Gaussian)
    blur_moderate = cv2.GaussianBlur(img, (9, 9), 3)
    cv2.imwrite(os.path.join(output_dir, "blur_moderate.jpg"), blur_moderate)
    
    # 3. High blur (Gaussian)
    blur_high = cv2.GaussianBlur(img, (15, 15), 5)
    cv2.imwrite(os.path.join(output_dir, "blur_high.jpg"), blur_high)
    
    # Create a comparison image
    # Stack images horizontally
    top_row = np.hstack((img, blur_low))
    bottom_row = np.hstack((blur_moderate, blur_high))
    combined = np.vstack((top_row, bottom_row))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    
    # Add labels to the combined image
    labels = ['Original', 'Low Blur', 'Moderate Blur', 'High Blur']
    positions = [(50, 50), (img.shape[1] + 50, 50), 
                (50, img.shape[0] + 50), (img.shape[1] + 50, img.shape[0] + 50)]
    
    for label, pos in zip(labels, positions):
        cv2.putText(combined, label, pos, font, font_scale, font_color, thickness)
    
    # Save the comparison image
    cv2.imwrite(os.path.join(output_dir, "comparison.jpg"), combined)
    
    print(f"Generated blurred images have been saved to {output_dir}")
    return {
        'original': img,
        'low_blur': blur_low,
        'moderate_blur': blur_moderate,
        'high_blur': blur_high,
        'comparison': combined
    }

def show_images(images_dict):
    """
    Display all images in a window
    """
    # Create windows for each image
    for name, img in images_dict.items():
        cv2.imshow(name, img)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Generate blurred images
        results = create_blurred_versions()
        
        # Show the images (optional)
        show_images(results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")