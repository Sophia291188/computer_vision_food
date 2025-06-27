
# 🍽️ Food-101: Calorie Estimation and Classification

A computer vision system for automatic food classification and calorie estimation, powered by deep learning and GPT-based feedback.

This project leverages a fine-tuned **ResNet-50** model trained on the **Food-101** dataset to classify food images into 101 categories and estimate their calorie content. It also features **multi-image intake tracking** and **GPT-powered dietary suggestions**, combining AI-based nutrition analysis with an interactive user experience.

---

## 🌟 Features

- 🥗 **Food Classification**: Classifies food images into 101 categories using ResNet-50.
- 🔢 **Calorie Estimation**: Estimates calories based on a lookup table (Excel or auto-generated CSV fallback).
- 🧠 **GPT-Based Suggestions**: Summarizes nutritional intake with intelligent feedback (e.g., fat loss tips, dietary advice).
- 📊 **Training Pipeline**: Full model training script with validation, checkpointing, and augmentation.
- 🖼️ **Gradio Interface**: Drag-and-drop interface for real-time prediction and calorie display.
- 🧾 **Multi-Image Intake Tracker**: Tracks multiple meals and provides a holistic summary.
- ⚡ **GPU Acceleration**: Compatible with Google Colab and CUDA-enabled devices.

---

## 📦 Project Overview

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `food_train_code.ipynb` | 🎯 Main training notebook with augmentation, validation, and checkpointing.  |
| `multi_food_intake_gpt_suggeste.ipynb` | 🤖 Upload multiple images and receive GPT-based nutrition suggestions. |
| `calorie_lookup_table.xlsx` | 🧮 Lookup table for calories per food class (customizable).                |
| `models/`              | 💾 Trained ResNet-50 models stored via Git LFS.                              |
| `.gitattributes`       | ⚙️ Git LFS config file for tracking model weights.                           |
| `README.md`            | 📘 This documentation file.                                                  |

> 📌 Notebooks are modular — you can train the model and run predictions separately.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install torch torchvision pandas gradio openai
```

You’ll also need an [OpenAI API key](https://platform.openai.com/account/api-keys) for GPT-based feedback (optional but recommended).

---

### 📁 Folder Structure

```
computer_vision_food/
├── models/                          # ResNet-50 model weights (.pth via Git LFS)
├── calorie_lookup_table.xlsx       # Food-to-calorie reference (Excel)
├── food_train_code.ipynb           # Training script
├── multi_food_intake_gpt_suggeste.ipynb  # Prediction + GPT feedback
├── computer_vision_06242024.ipynb  # Optional Colab setup
├── .gitattributes                  # Git LFS configuration
└── README.md                       # Project documentation
```

---

## 🧠 Model Training Instructions

1. **Open** `food_train_code.ipynb`
2. **Steps performed in the notebook**:
   - Load and extract Food-101 dataset
   - Split into train/val/test
   - Normalize and augment images
   - Initialize ImageNet-pretrained ResNet-50
   - Train model and save best checkpoint to `models/`

> 📝 Skip training if you use the pretrained model directly from the `models/` folder.

---

## 🎯 GPT-Enhanced Food Intake Feedback

To receive intelligent summaries based on your meals:

1. **Open** `multi_food_intake_gpt_suggeste.ipynb`
2. Upload multiple meal images (breakfast, lunch, dinner, etc.)
3. Model will:
   - Predict each food image
   - Match calories using the lookup table
   - Display total calorie intake
   - Ask GPT to summarize the nutritional balance

Example GPT summary:
> 🧠 *"You consumed high-carb meals today with little protein. Consider adding more vegetables and lean protein sources tomorrow."*

> 🔐 Don’t forget to set your OpenAI API key in the notebook.

---

## 🧪 Model Details

| Item             | Value                        |
|------------------|------------------------------|
| Architecture     | ResNet-50                    |
| Pretrained       | ImageNet                     |
| Loss Function    | CrossEntropyLoss             |
| Optimizer        | Adam                         |
| LR Scheduler     | ReduceLROnPlateau            |
| Accuracy         | ~78–80% Top-1 on validation  |
| Training Time    | ~2–3 hours (Colab T4 GPU)    |

---

## 🖼️ Using the Gradio Web Demo

> Launches automatically at the end of `multi_food_intake_gpt_suggeste.ipynb`

- Drag and drop a food image
- Get predicted label + confidence
- See estimated calories
- Upload more images for multi-meal tracking
- Get GPT nutrition summary

---

## 💡 Use Cases

- 📱 Nutrition tracking mobile/web demo
- 🧮 Visual calorie calculator
- 📒 AI-powered food journaling
- 🧑‍⚕️ Dietary recommendation system for personal use or clinics

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Free to use for personal, academic, or demo purposes.

---

## 🙏 Acknowledgements

- [Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) by ETH Zurich  
- [PyTorch](https://pytorch.org/) for model training  
- [Gradio](https://gradio.app/) for interactive UI  
- [OpenAI](https://openai.com/) for GPT-4 feedback generation

---

## 📬 Contact

For questions, suggestions, or collaborations:  
**Sophia Liu (Shu Yu Liu)**  
GitHub: [@Sophia291188](https://github.com/Sophia291188)
