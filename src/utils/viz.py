import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(image):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()


def show_tensor(image):
    """
    Display a tensor image using matplotlib.
    The tensor is expected to be in the format (C, H, W) with values in the range [0, 1].
    """
    assert image.ndim == 3 and image.shape[0] == 3, f"Expected (C, H, W), got {image.shape}"

    # Convert tensor to numpy array and permute dimensions to (H, W, C)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')  # Hide the axis
    plt.show()


def colored_img_torch(image):
    """Convert a (3, H, W) tensor with values 0-8 to an RGB image."""
    assert image.ndim == 3 and image.shape[0] == 3, f"Expected (3, H, W), got {image.shape}"

    # Define the color palette
    palette = np.array([
        [0, 0, 0],  # 0: Black
        [244, 13, 61],  # 1: Red
        [51, 221, 255],  # 2: Cyan
        [250, 50, 83],  # 3: Pink
        [28, 230, 141],  # 4: Green
        [221, 255, 51],  # 5: Yellow
        [13, 13, 242],  # 6: Blue
        [242, 185, 14],  # 7: Orange
        [184, 61, 245],  # 8: Purple
        [242, 185, 200]  # 9: Light Pink (Extra)
    ], dtype=np.uint8)

    # Convert to NumPy
    image_np = image.cpu().numpy().astype(np.uint8)  # Shape: (3, H, W)

    # Apply palette to each channel separately
    color_channels = [palette[image_np[i]] for i in range(3)]  # List of (H, W, 3)

    # Compute final RGB image (mean across 3 channels)
    output_img = np.mean(color_channels, axis=0).astype(np.uint8)  # Shape: (H, W, 3)

    return output_img


def colored_img(image):
    output_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    unique_values = np.unique(image)
    palette = [
        [0, 0, 0],
        [244, 13, 61],
        [51, 221, 255],
        [250, 50, 83],
        [28, 230, 141],
        [221, 255, 51],
        [13, 13, 242],
        [242, 185, 14],
        [184, 61, 245],
        [242, 185, 200]
    ]
    assert len(unique_values) <= len(palette), f"Got {len(unique_values)} unique values: {unique_values}"
    for value in unique_values:
        if 0 <= value < len(palette):
            mask = (image == value)
            output_img[mask] = palette[value]
    return output_img


def overlay_img(image, ann, alpha=0.3):
    overlay_image = image.copy()
    mask = ann != [0, 0, 0]  # Exclude background (black)
    overlay_image[mask.all(axis=2)] = cv2.addWeighted(
        image[mask.all(axis=2)], 1 - alpha,
        ann[mask.all(axis=2)], alpha,
        0)
    return overlay_image
