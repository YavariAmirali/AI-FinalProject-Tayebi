import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import math

# --- SETUP PATHS ---
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

# Adjust these if your folder structure is different
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
csv_path = os.path.join(RESULTS_DIR, 'error_analysis.csv')


def show_fp_images():
    # Load the dataframe
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Filter for False Positives
    fp_data = df[df['Type'] == 'False Positive']

    num_images = len(fp_data)
    print(f"📂 Searching for images in: {TEST_DIR}")
    print(f"📊 Found {num_images} False Positive images in CSV.")

    if num_images == 0:
        print("✅ No False Positive images found to plot.")
        return

    # Setup the grid for plotting
    cols = 5
    rows = math.ceil(num_images / cols)

    # Create figure (size adjusts dynamically)
    plt.figure(figsize=(20, 4 * rows))

    # Iterate and plot
    for idx, (index, row) in enumerate(fp_data.iterrows()):
        filename = row['Filename']
        true_label = row['True_Label']

        # Use 'Confidence' instead of 'Model_Confidence'
        conf = row['Confidence']

        # Construct path: data/test/NORMAL/image.jpg
        full_image_path = os.path.join(TEST_DIR, true_label, filename)

        # Plotting
        plt.subplot(rows, cols, idx + 1)

        if os.path.exists(full_image_path):
            img = cv2.imread(full_image_path)
            if img is not None:
                # Convert BGR to Grayscale for X-Ray visualization
                plt.imshow(img, cmap='gray')
            else:
                plt.text(0.5, 0.5, 'Corrupt Image', ha='center', va='center')
        else:
            plt.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            if idx < 3:  # Only print first few missing file warnings
                print(f"⚠️ Warning: File not found at {full_image_path}")

        # Add title
        plt.title(f"{filename}\nConf: {conf:.4f}", fontsize=9)
        plt.axis('off')

    plt.suptitle(f"False Positives Analysis ({num_images} Images)", fontsize=16, y=1.02)
    plt.tight_layout()

    # --- SAVE THE IMAGE ---
    save_path = os.path.join(RESULTS_DIR, 'false_positive_summary.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Successfully saved summary image to: {save_path}")

    plt.show()


if __name__ == "__main__":
    show_fp_images()