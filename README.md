# **LLM-Text-Classification: Fine-Tuning OPT-125M for Sentiment Analysis**  

## **Overview**  
This project fine-tunes **Meta’s OPT-125M** model for **text classification**, specifically **sentiment analysis**. The pipeline is built using **PyTorch Lightning**, leveraging:  

- **Large-scale fine-tuning** on 45K training samples  
- **W&B integration** for real-time experiment tracking  
- **Early stopping & checkpointing** for stability  
- **Gradient accumulation & mixed precision** for efficient training  

This project explores **LLM-based text classification performance and scalability** with real-world datasets.  

---

## **🚀 Key Features**  
✅ **Fine-tunes OPT-125M** on a large-scale sentiment dataset  
✅ **PyTorch Lightning pipeline** for modular training  
✅ **W&B logging** for tracking loss, accuracy, and model checkpoints  
✅ **Early stopping** to prevent overfitting  
✅ **Gradient accumulation** for handling large batches on limited GPU memory  
✅ **Multi-GPU training support**  

---

## **📁 Project Structure**  

```bash
├── llm-text-classification/
│   ├── main.py                   # Training pipeline
│   ├── config.py                 # Training hyperparameters
│   ├── dataloader.py             # Dataset loading module
│   ├── lightning_trainer.py      # PyTorch Lightning trainer
│   ├── lightning_model_utils.py  # Model setup utilities
│   ├── lightning_inference.py    # Model inference module
│   ├── README.md                 # Project documentation (this file)
```

---

## **🛠️ Setup & Training**  

### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Run Fine-Tuning**  
```bash
python main.py
```

💡 **To train on a multi-GPU setup:**  
Modify `config.py` to use **multiple devices** and increase `batch_size`.  

---

## **📊 Evaluation & Results**  

| Model  | Dataset  | Accuracy | F1 Score |  
|--------|---------|----------|----------|  
| OPT-125M | Custom Sentiment Dataset | **85%** | **0.87** |  

✔️ **Final Performance:** Achieved **85% accuracy** on sentiment classification  
✔️ **Challenges:** Handling **large-scale fine-tuning on limited GPU memory**  
✔️ **Solution:** Used **gradient accumulation & mixed precision**  

---

## **🔎 Inference (Classify Sentiment)**  

```bash
python lightning_inference.py --text "This movie was amazing!"
```
Example Output:  
```json
{
  "text": "This movie was amazing!",
  "predicted_label": "positive"
}
```

---

## **🔮 Next Steps**  
🔹 **Experiment with larger models (OPT-350M, OPT-1.3B)**  
🔹 **Expand to multi-class classification (e.g., emotions, topics)**  
🔹 **Deploy as an API for real-time classification**  

---

## **📌 Why This Project Matters?**  
This project **demonstrates LLM fine-tuning for text classification**, solving:  
- **Handling large-scale datasets efficiently**  
- **Real-time experiment tracking with W&B**  
- **Optimized training on resource-constrained GPUs**  

📌 **Ideal for:** AI/ML roles focusing on **LLM fine-tuning, model evaluation, and scalable NLP applications.**  
