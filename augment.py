import os
import cv2
import numpy as np

# Configuration
NAME = "you"
DATA_DIR = "data"
SOURCE_DIR = os.path.join(DATA_DIR, NAME)

# Augmentation parameters
ROTATION_ANGLES = [-15, 15]  # degrees
BRIGHTNESS_FACTORS = [0.7, 1.3]  # dimmer and brighter

def rotate(image, angle):
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def augment_image(img, base_name, count):
    augmented = []
    # Rotation
    for angle in ROTATION_ANGLES:
        rotated = rotate(img, angle)
        augmented.append((rotated, f"{base_name}_rot{angle}_{count}.jpg"))
    # Brightness
    for factor in BRIGHTNESS_FACTORS:
        bright = adjust_brightness(img, factor)
        augmented.append((bright, f"{base_name}_bright{int(factor*100)}_{count}.jpg"))
    # Flip
    flipped = cv2.flip(img, 1)
    augmented.append((flipped, f"{base_name}_flip_{count}.jpg"))
    return augmented

def augment_all():
    if not os.path.isdir(SOURCE_DIR):
        print(f"❌ Folder not found: {SOURCE_DIR}")
        return

    count = 0
    for img_name in sorted(os.listdir(SOURCE_DIR)):
        img_path = os.path.join(SOURCE_DIR, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        base_name = os.path.splitext(img_name)[0]
        new_images = augment_image(img, base_name, count)

        for new_img, filename in new_images:
            save_path = os.path.join(SOURCE_DIR, filename)
            cv2.imwrite(save_path, new_img)
            print(f"✅ Saved: {filename}")

        count += 1

    print("\n🎉 Augmentation complete.")

if __name__ == "__main__":
    augment_all()