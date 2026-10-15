# 💬 ToneLens AI — Message Sentiment Analyzer

ToneLens AI is an AI-powered message sentiment analysis application that classifies text as **Positive** or **Negative** using a pretrained DistilBERT model. It provides confidence scores to help users understand the general sentiment expressed in a message.

The project combines a user-friendly interface, transformer-based NLP, MLflow experiment tracking, and deployment through Hugging Face Spaces.

## ✨ Features

- **Sentiment Analysis:** Classifies messages as positive or negative.
- **Confidence Score:** Displays the model's confidence for each prediction.
- **Interactive UI:** Provides a clean, responsive interface with a dark theme.
- **MLflow Tracking:** Logs prediction confidence and sentiment metadata for local app predictions.
- **Model Evaluation:** Supports tracking evaluation metrics such as accuracy, precision, and recall.
- **Hugging Face Deployment:** Makes the application accessible through a public Space.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python, JavaScript |
| AI / NLP | Hugging Face Transformers, DistilBERT |
| Interface | Gradio, HTML, CSS |
| Experiment Tracking | MLflow |
| Model Hosting | Hugging Face Spaces |
| Version Control | Git, GitHub |

## 🧠 Model

ToneLens AI uses a pretrained DistilBERT sentiment classification model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.

**Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

The model predicts a sentiment label and confidence score. Predictions are estimates and may not fully capture sarcasm, context, or a sender's true intention.

## 🚀 Live Demo

Try the deployed application on Hugging Face Spaces:

**[Open ToneLens AI](https://huggingface.co/spaces/diviii15/Tonelens)**

## 💻 Run Locally

### Prerequisites

- Python installed
- Git installed
- Internet connection for downloading the pretrained model and dependencies

### 1. Clone the repository

```bash
git clone https://github.com/Divya071506/ToneLens-AI.git
cd ToneLens-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install gradio transformers torch mlflow
```

### 4. Start the MLflow tracking server

Open a separate terminal, activate the same environment, and run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open the MLflow dashboard:

`http://127.0.0.1:5000`

### 5. Run the application

In another terminal, activate the virtual environment and run:

```bash
python app.py
```

Open the local URL printed by Gradio, usually:

`http://127.0.0.1:7860`

Enter a message and click **Analyze sentiment** to view the prediction.

## 📊 MLflow Experiment Tracking

The local application logs prediction information to the `ToneLens-AI-Experiment` experiment.

Tracked information includes:

- **Parameter:** Model identifier
- **Metric:** Prediction confidence
- **Tags:** Predicted sentiment and event type

The MLflow dashboard allows you to inspect individual prediction runs and review model evaluation metrics logged separately.

> Note: The local Gradio application logs directly to the local MLflow server. Predictions made on the public Hugging Face Space are not automatically sent to your local MLflow instance.

## 📁 Project Structure

```text
ToneLens-AI/
├── app.py
├── tracking_api.py
├── mlflow_demo.py
├── .gitignore
└── README.md
```

- `app.py` — Local sentiment analysis interface and MLflow logging.
- `tracking_api.py` — FastAPI endpoint for prediction tracking.
- `mlflow_demo.py` — MLflow demonstration and/or evaluation script.
- `.gitignore` — Excludes local environment files and generated tracking data.

## 🔮 Future Enhancements

- Extend sentiment analysis to more nuanced emotions.
- Improve handling of sarcasm and context.
- Add richer experiment comparisons and model evaluation visualizations.
- Integrate secure remote prediction tracking for the deployed application.

