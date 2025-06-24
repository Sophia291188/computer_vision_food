# computer_vision_food
# Food-101 Food Recognition and Calorie Estimation Demo

This project provides a comprehensive solution for food image recognition and calorie estimation using a fine-tuned ResNet-50 model on the Food-101 dataset, presented via an interactive Gradio web interface.

## 🌟 Features

* **Food Image Recognition**: Identifies 101 different food categories from images.
* **Calorie Estimation**: Provides an estimated calorie count for the recognized food item based on a lookup table.
* **Pre-trained Model**: Leverages the powerful ResNet-50 architecture with pre-trained ImageNet weights for efficient learning.
* **Data Augmentation**: Employs robust data augmentation techniques during training to improve model generalization.
* **Checkpointing**: Supports saving and loading model training checkpoints for seamless resumption.
* **Gradio Web Interface**: An easy-to-use drag-and-drop web interface for real-time predictions.

## 🚀 Getting Started

This project is designed to be run primarily in a Google Colab environment due to its reliance on GPU acceleration and direct integration with Google Drive.

### Prerequisites

* Google Account (for Colab and Google Drive)
* Basic understanding of Python and Deep Learning concepts.

### Setup and Running the Notebook

1.  **Open in Google Colab**:
    * Upload the provided `.ipynb` file to your Google Drive, or open it directly in Colab.
    * Ensure your Colab runtime is set to **GPU** (Runtime -> Change runtime type -> Hardware accelerator -> GPU).

2.  **Run All Cells**:
    * Execute all cells in the Colab notebook sequentially (Runtime -> Run all).

    The notebook will perform the following steps:
    * **Mount Google Drive**: Connects to your Google Drive to store/load models and data.
    * **Download & Extract Food-101 Dataset**: Downloads the ~5GB Food-101 dataset and extracts it to `/content/food-101` (Colab's fast local storage).
    * **Load Calorie Table**: Attempts to load `calorie_lookup_table.xlsx` from your Google Drive. If not found or if there's an error, it will generate a random one and save it as `food101_random_calorie_table.csv`.
    * **Data Preparation**: Splits the Food-101 dataset into 80% training, 10% validation, and 10% test sets, applying appropriate transformations.
    * **Model Initialization & Training**: Initializes a ResNet-50 model, potentially loads a pre-trained checkpoint, and starts/resumes training for a specified number of epochs. The best model based on validation accuracy is saved to `food101_resnet50.pth` in your Google Drive.
    * **Final Test Set Evaluation**: Evaluates the best-saved model on the unseen test set to report final performance metrics.
    * **Launch Gradio UI**: Starts the interactive web demo.

3.  **Access the Gradio Demo**:
    * After the last cell finishes executing, a public URL (e.g., `https://xxxxxx.gradio.live`) will be printed in the Colab output.
    * Click this link to open the Gradio web interface in your browser.

## 📸 Using the Gradio Demo

1.  **Upload Image**:
    * In the Gradio interface, click the "Upload Food Image" area.
    * You can drag and drop an image file, or click to browse and select an image from your local computer.
    * **(Note)** The current Gradio UI in this code uses `gr.File(type="filepath")` which means you'll select a file from your disk. If you previously had "webcam" input, that functionality has been removed in the provided code.
2.  **Submit for Prediction**:
    * Once the image is uploaded, click the "Submit" button.
3.  **View Results**:
    * The interface will display the predicted food class, confidence level (percentage), and an estimated calorie count.

## 📂 Project Structure (on Google Drive)

Upon running the notebook, the `food101_project` folder in your Google Drive (`/content/drive/MyDrive/food101_project`) will contain:

* `food-101.tar.gz`: The downloaded compressed Food-101 dataset.
* `calorie_lookup_table.xlsx` (Optional): If you provide this file, it will be used for calorie lookups.
* `food101_random_calorie_table.csv`: A randomly generated calorie table if the Excel file is not found.
* `food101_resnet50.pth`: The best-performing trained model checkpoint.
* `food101_resnet50_epoch_X_acc_YY.YY_loss_ZZ.ZZ.pth`: Snapshots of the model after each epoch of training.

## 🧪 Model Details

* **Architecture**: ResNet-50
* **Pre-training**: ImageNet weights (`models.ResNet50_Weights.DEFAULT`)
* **Loss Function**: Cross-Entropy Loss
* **Optimizer**: Adam
* **Learning Rate Scheduler**: `ReduceLROnPlateau` to dynamically adjust learning rate based on validation loss.

## License

This project is open-source and available under the [MIT License](LICENSE).
(Note: You might need to create a LICENSE file in your GitHub repo if you want to explicitly state the license.)

## 🙏 Acknowledgements

* **Food-101 Dataset**: Provided by ETH Zurich.
* **PyTorch**: For the deep learning framework.
* **Gradio**: For easily building the web demo.
