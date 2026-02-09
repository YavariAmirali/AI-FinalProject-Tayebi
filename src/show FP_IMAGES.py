import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import math

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(DATA_DIR, 'test')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
csv_path = os.path.join(RESULTS_DIR, 'error_analysis.csv')
image_folder_path = TEST_DIR



def show_fp_images():
    # 1. Load the dataframe
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path)

    # 2. Filter for False Positives
    fp_data = df[df['Error_Type'] == 'False Positive']

    num_images = len(fp_data)
    print(f"Path configured as: {image_folder_path}")
    print(f"Found {num_images} False Positive images in CSV.")

    if num_images == 0:
        print("No False Positive images found to plot.")
        return

    # 3. Setup the grid for plotting
    cols = 5
    rows = math.ceil(num_images / cols)

    # Set figure size (width=20, height adjusts based on rows)
    plt.figure(figsize=(20, 4 * rows))

    # 4. Iterate and plot
    for idx, (index, row) in enumerate(fp_data.iterrows()):
        filename = row['Filename']
        true_label = row['True_Label']
        path_with_subfolder = os.path.join(image_folder_path, true_label, filename)
        path_flat = os.path.join(image_folder_path, filename)

        if os.path.exists(path_with_subfolder):
            full_image_path = path_with_subfolder
        elif os.path.exists(path_flat):
            full_image_path = path_flat
        else:
            full_image_path = path_with_subfolder

        plt.subplot(rows, cols, idx + 1)

        if os.path.exists(full_image_path):
            img = cv2.imread(full_image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img, cmap='gray')
            else:
                plt.text(0.5, 0.5, 'Corrupt Image', ha='center', va='center')
        else:
            plt.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            if idx < 3:
                print(f"Warning: File not found at {full_image_path}")

        # Add title with Filename and Confidence
        title_text = f"{filename}\nTrue: {true_label} | Conf: {row['Model_Confidence']:.4f}"
        plt.title(title_text, fontsize=8)
        plt.axis('off')

    # --- SAVE THE IMAGE ---
    save_path = os.path.join(RESULTS_DIR, 'false_positive_summary.png')
    plt.savefig(save_path)
    print(f"Successfully saved image to: {save_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_fp_images()