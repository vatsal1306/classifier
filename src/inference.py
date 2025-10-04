import argparse
import os

import torch
import webdataset as wds
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import config as default_config
from src.data.transformations import get_test_transforms  # For single image inference
from src.models import get_model


def visualize_mistakes(mistakes, output_path, grid_size=5):
    """
    Creates and saves a collage of the most confident misclassified images.
    """
    if not mistakes:
        print("No mistakes to visualize.")
        return

    # Sort mistakes by confidence (descending)
    mistakes.sort(key=lambda x: x[2], reverse=True)

    num_images = min(len(mistakes), grid_size * grid_size)

    # Create a grid to hold the images
    img_size = mistakes[0][0].size[0]
    grid_img_size = grid_size * img_size
    grid_image = Image.new('RGB', (grid_img_size, grid_img_size), 'white')
    draw = ImageDraw.Draw(grid_image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for i in range(num_images):
        image, true_label, confidence = mistakes[i]

        row = i // grid_size
        col = i % grid_size

        # Paste image onto the grid
        grid_image.paste(image, (col * img_size, row * img_size))

        # Add text
        true_text = "True: Human" if true_label == 1 else "True: Non-Human"
        pred_text = "Pred: Non-Human" if true_label == 1 else "Pred: Human"
        conf_text = f"Conf: {confidence:.2f}"

        draw.text((col * img_size + 5, row * img_size + 5), true_text, fill="red", font=font)
        draw.text((col * img_size + 5, row * img_size + 20), pred_text, fill="red", font=font)
        draw.text((col * img_size + 5, row * img_size + 35), conf_text, fill="red", font=font)

    grid_image.save(output_path)
    print(f"Mistakes collage saved to {output_path}")


def run_inference(checkpoint_path, data_dir, output_dir):
    """
    Runs inference on the test set, calculates accuracy, and visualizes mistakes.
    """
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model from {checkpoint_path}")
    model = get_model(default_config.MODEL_NAME, pretrained=False, num_classes=default_config.OUTPUT_FEATURES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Dataloader ---
    # We create a temporary config to point the dataloader to the right place
    class InferenceConfig:
        DATA_DIR = data_dir
        TEST_BATCH_SIZE = default_config.TEST_BATCH_SIZE

    print("Building test dataloader...")
    # This requires a custom WebDataset pipeline for inference to get original images
    test_urls = [
        os.path.join(data_dir, "test/human", "shard-*.tar"),
        os.path.join(data_dir, "test/non_human", "shard-*.tar")
    ]

    # Create a pipeline that keeps the original image for visualization
    preproc = get_test_transforms()
    dataset = (
        wds.WebDataset(test_urls)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map_tuple(lambda img: (preproc(img), img), lambda x: torch.tensor(int(x.decode())))
        # (transformed_img, original_img), label
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=InferenceConfig.TEST_BATCH_SIZE, num_workers=4)

    # --- Inference Loop ---
    correct_predictions = 0
    total_samples = 0
    mistakes = []

    print("Running inference...")
    with torch.no_grad():
        for (transformed_images, original_images), labels in tqdm(dataloader, desc="Inference"):
            transformed_images = transformed_images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(transformed_images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Find mistakes
            misclassified_mask = (preds != labels).squeeze()
            if misclassified_mask.any():
                misclassified_imgs = [img for i, img in enumerate(original_images) if misclassified_mask[i]]
                misclassified_labels = labels[misclassified_mask]
                misclassified_probs = probs[misclassified_mask]

                for img, true_label, prob in zip(misclassified_imgs, misclassified_labels, misclassified_probs):
                    confidence = prob.item() if pred == 1 else 1 - prob.item()
                    mistakes.append((img, int(true_label.item()), confidence))

    # --- Results ---
    accuracy = correct_predictions / total_samples
    print(f"\n--- Inference Complete ---")
    print(f"Accuracy on Test Set: {accuracy:.4f}")

    # --- Visualize ---
    visualize_mistakes(mistakes, os.path.join(output_dir, "worst_mistakes.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and visualize model mistakes.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--data_dir", type=str, default=default_config.DATA_DIR,
                        help="Path to the processed data directory.")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save inference results.")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.data_dir, args.output_dir)
