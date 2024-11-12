from PIL import Image
import os

def convert_to_grayscale(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    grayscale_img = img.convert("L")
    
    # Get the original filename and extension
    base, ext = os.path.splitext(image_path)
    
    # Create a new filename with the '-gray' postfix
    new_filename = f"{base}-gray{ext}"
    
    # Save the grayscale image with the new filename
    grayscale_img.save(new_filename)
    
    print(f"Grayscale image saved as {new_filename}")


if __name__ == "__main__":
    for filename in os.listdir():
        if filename.endswith(".png"):
            convert_to_grayscale(filename)