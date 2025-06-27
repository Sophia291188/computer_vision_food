
# 🍽️ computer_vision_food

A comprehensive food image recognition and calorie estimation system powered by a fine-tuned **ResNet-50** model on the **Food-101** dataset. It also supports **multi-image food intake tracking** and **GPT-based dietary feedback**.

This project combines computer vision with nutritional intelligence, offering an end-to-end solution from training to inference and suggestion generation.

---

## 📦 Overview

- **ResNet-50 model** fine-tuned on Food-101 dataset
- **Gradio web UI** for real-time food prediction & calorie estimation
- **GPT-4 powered dietary summary** for multiple meals
- Modular Jupyter Notebooks for training and deployment
- Model weights managed via **Git LFS**
- Compatible with Google Colab (optional)

---

## 📁 File Descriptions

| File / Folder | Description |
|---------------|-------------|
| `food_train_code.ipynb` | ✅ Main training notebook for ResNet-50 on Food-101. Includes data loading, augmentation, training loop, validation, checkpoint saving. |
| `multi_food_intake_gpt_suggeste.ipynb` | 🍱 Upload multiple food images and get GPT-generated suggestions based on predicted food types and calorie intake. |
| `calorie_lookup_table.xlsx` | 📊 A calorie reference table for 101 food classes. Used for mapping class predictions to estimated calories. |
| `computer_vision_06242024.ipynb` | (Optional) Initial project setup notebook via Google Colab, not required for usage. |
| `models/` | 📁 Folder containing the best model weights (`food101_resnet50.pth`). Uses Git LFS for versioning. |
| `.gitattributes` | Configuration for Git LFS to track large model files. |
| `README.md` | 📖 Project documentation (this file). |

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- Gradio
- openai (if using GPT suggestions)

```bash
pip install torch torchvision pandas gradio openai
```

---

## 🧠 Model Training

To train the model from scratch or resume training:

1. **Open** `food_train_code.ipynb`
2. The notebook will:
   - Load and preprocess Food-101 dataset
   - Split into training, validation, and test sets
   - Apply augmentation and normalization
   - Initialize ResNet-50 (ImageNet pre-trained)
   - Train and validate the model for each epoch
   - Save best model as `food101_resnet50.pth`

> 💡 You can skip training and use the provided pretrained model in `models/` folder.

---

## 🤖 Gradio + GPT-Based Prediction

To run inference and generate dietary feedback:

1. **Open** `multi_food_intake_gpt_suggeste.ipynb`
2. Upload multiple food images (breakfast, lunch, dinner, etc.)
3. The notebook will:
   - Load the saved `food101_resnet50.pth`
   - Predict food class and match calorie from table
   - Summarize total calorie intake
   - Use **OpenAI GPT-4** to generate a personalized summary (e.g., "You ate high-carb meals today. Consider more vegetables tomorrow.")
4. Requires OpenAI API key (insert in code cell)

> 🔐 GPT usage is optional but recommended for intelligent summaries.

---

## 🧪 Model Details

- **Architecture**: ResNet-50
- **Pretrained**: Yes (ImageNet weights)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Accuracy**: Achieves ~78–80% Top-1 accuracy on Food-101 validation set

---

## 🎯 Use Cases

- Nutrition tracking app demo
- Smart calorie calculator based on image input
- GPT-integrated food journaling system
- AI-powered diet monitoring and coaching

---

## 📂 Folder Structure

```
computer_vision_food/
│
├── models/                      # Trained ResNet-50 model weights (.pth via LFS)
├── calorie_lookup_table.xlsx   # Excel file with food-calorie mapping
├── food_train_code.ipynb       # Training pipeline
├── multi_food_intake_gpt_suggeste.ipynb  # Inference + GPT suggestions
├── computer_vision_06242024.ipynb        # Initial Colab setup (optional)
├── .gitattributes              # Git LFS configuration
└── README.md                   # Project documentation
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use and adapt it for your own research, demo, or development purposes.

---

## 🙏 Acknowledgements

- [Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) by ETH Zurich  
- [PyTorch](https://pytorch.org/) for training and inference  
- [Gradio](https://gradio.app/) for UI  
- [OpenAI](https://openai.com/) for GPT-based summaries

---

## 📬 Contact

For questions or collaboration requests:  
**Sophia Liu (Shu Yu Liu)** — [GitHub Profile](https://github.com/Sophia291188)
