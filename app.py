
import html
import gradio as gr
import mlflow
from transformers import pipeline

# ---------------- MODEL ----------------

print("Loading ToneLens AI model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

print("ToneLens AI is ready!")

# ---------------- MLFLOW ----------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ToneLens-AI-Experiment")


# ---------------- ANALYSIS ----------------

def analyze_message(message):
    if not message or not message.strip():
        return "", "Please enter a message first."

    try:
        prediction = classifier(message[:512])[0]
        label = prediction["label"].upper()
        confidence = float(prediction["score"])
        percent = confidence * 100

        positive = label == "POSITIVE"
        sentiment = "Positive 😊" if positive else "Negative 😟"

        explanation = (
            "The model identifies a generally positive sentiment "
            "in this message."
            if positive
            else
            "The model identifies a generally negative sentiment "
            "in this message."
        )

        # Log each prediction directly to MLflow.
        tracking_message = ""

        try:
            with mlflow.start_run(run_name="ToneLens-Prediction"):
                mlflow.log_param("model", "distilbert-sst2")
                mlflow.log_metric("prediction_confidence", confidence)
                mlflow.set_tag("predicted_sentiment", label)
                mlflow.set_tag("event_type", "user_prediction")

            print(f"MLflow logged: {label}, confidence={confidence:.4f}")
            tracking_message = "Prediction tracked in MLflow."

        except Exception as error:
            print(f"MLflow logging failed: {error}")
            tracking_message = "Prediction shown, but MLflow logging failed."

        # Only display fixed model labels and numeric confidence.
        result_html = f"""
        <div class="result-card">
            <p class="eyebrow">ANALYSIS RESULT</p>
            <h2>{sentiment}</h2>

            <div class="confidence-row">
                <span>Model confidence</span>
                <strong>{percent:.2f}%</strong>
            </div>

            <div class="meter">
                <div class="meter-fill" style="width: {percent:.2f}%"></div>
            </div>

            <p class="explanation">{explanation}</p>
        </div>
        """

        return result_html, f"Analysis complete! {tracking_message}"

    except Exception as error:
        print(f"Analysis error: {error}")
        return "", "Analysis failed. Please try a shorter message."


# ---------------- CUSTOM DESIGN ----------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background:
        radial-gradient(ellipse at 50% 0%, #242044 0%, transparent 48%),
        #090b12 !important;
    color: #f2f2fa !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container, .gradio-container * {
    font-family: 'DM Sans', sans-serif;
}

body {
    background: #090b12 !important;
}

#tone-header {
    text-align: center;
    padding: 42px 10px 28px;
}

.logo {
    display: grid;
    place-items: center;
    width: 56px;
    height: 56px;
    margin: 0 auto 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #9d7bff, #5b8cff);
    box-shadow: 0 8px 35px #8065ff40;
    color: white;
    font: 700 22px 'Space Grotesk', sans-serif;
}

.eyebrow {
    color: #a997ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin: 0 0 12px;
}

#tone-header h1 {
    font: 700 clamp(38px, 7vw, 58px) 'Space Grotesk', sans-serif;
    letter-spacing: -2px;
    margin: 8px 0 12px;
    color: #f2f2fa;
}

#tone-header h1 span {
    color: #a997ff;
}

.subtitle {
    max-width: 460px;
    margin: 0 auto;
    color: #a4a6b9;
    line-height: 1.8;
    font-size: 15px;
}

#tone-panel {
    border: 1px solid #292a40 !important;
    border-radius: 24px !important;
    padding: clamp(20px, 5vw, 34px) !important;
    background: linear-gradient(145deg, #171827, #10121c) !important;
    box-shadow: 0 25px 80px #0005 !important;
}

#tone-panel label {
    color: #f2f2fa !important;
    font-weight: 600 !important;
}

#tone-panel textarea {
    color: #f2f2fa !important;
    background: #0c0e17 !important;
    border: 1px solid #303146 !important;
    border-radius: 14px !important;
    line-height: 1.8 !important;
}

#tone-panel textarea:focus {
    border-color: #9580ff !important;
    box-shadow: 0 0 0 3px #9580ff1c !important;
}


#char-count {
    color: #85879c !important;
    font-size: 12px !important;
}

#status {
    color: #a4a6b9 !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    margin-top: 12px !important;
}


#analyze-btn {
    background: linear-gradient(110deg, #8869f7, #6488ff) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
    min-height: 46px;
}

#analyze-btn:hover {
    filter: brightness(1.1);
}



.result-card {
    margin-top: 12px;
    padding: 22px;
    border: 1px solid #34304f;
    border-radius: 17px;
    background: #121321;
    animation: appear 0.3s ease;
}

.result-card h2 {
    font: 600 27px 'Space Grotesk', sans-serif;
    color: #f2f2fa;
    margin: 12px 0 22px;
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: #b9bbcd;
    font-size: 13px;
}

.confidence-row strong {
    color: #c5b8ff;
}

.meter {
    height: 7px;
    background: #29283d;
    border-radius: 10px;
    overflow: hidden;
    margin: 12px 0 20px;
}

.meter-fill {
    height: 100%;
    background: linear-gradient(90deg, #8c70ff, #63a5ff);
    border-radius: inherit;
    transition: width 0.5s ease;
}

.explanation {
    color: #a9abc0;
    font-size: 13px;
    line-height: 1.8;
}

#tone-note {
    color: #77798e;
    font-size: 11px;
    line-height: 1.8;
    padding: 14px 4px 0;
}

#tone-footer {
    padding: 24px 0 30px;
    text-align: center;
    color: #62647a;
    font-size: 10px;
    letter-spacing: 1.5px;
}

@keyframes appear {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


# ---------------- INTERFACE ----------------

with gr.Blocks(
    title="ToneLens AI | Message Sentiment Analyzer",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate"
    )
) as demo:

    gr.HTML("""
        <header id="tone-header">
            <div class="logo">TL</div>
            <p class="eyebrow">AI-POWERED COMMUNICATION</p>
            <h1>ToneLens <span>AI</span></h1>
            <p class="subtitle">
                Understand the sentiment behind your words.
                Analyze a message instantly with AI.
            </p>
        </header>
    """)

    with gr.Column(elem_id="tone-panel"):

        message_input = gr.Textbox(
            label="Your message",
            placeholder=(
                "Type something like: "
                "Thank you so much for helping me!"
            ),
            lines=5,
            max_lines=10,
            elem_id="message-input"
        )

        with gr.Row():
            char_count = gr.Markdown(
                "0 characters",
                elem_id="char-count"
            )

            analyze_button = gr.Button(
                "Analyze sentiment →",
                variant="primary",
                elem_id="analyze-btn"
            )

        status = gr.Markdown(
            "AI model ready! Enter a message to begin.",
            elem_id="status"
        )

        result = gr.HTML(value="")

        gr.HTML("""
            <p id="tone-note">
                AI predictions are estimates and may not capture
                sarcasm, context, or the sender's true intention.
            </p>
        """)

    gr.HTML("""
        <footer id="tone-footer">
            TONELENS AI · BROWSER-INSPIRED SENTIMENT ANALYSIS
        </footer>
    """)

    message_input.change(
        fn=lambda message: f"{len(message or '')} characters",
        inputs=message_input,
        outputs=char_count
    )

    analyze_button.click(
        fn=analyze_message,
        inputs=message_input,
        outputs=[result, status]
    )


if __name__ == "__main__":
    demo.launch()
