import os
from flask import Flask, render_template, request, jsonify
from src.pipeline import SupportPipeline

app = Flask(__name__)

# Initialize and train pipeline once at app start
pipeline = SupportPipeline()
pipeline.train_pipeline(exclude_golden_from_retrieval=True)

SAMPLE_QUERIES = [
    {
        "label": "Software Update Bug",
        "text": "@AppleSupport My iPhone screen went completely black after the latest update and won't turn on!"
    },
    {
        "label": "Hardware / Battery Drain",
        "text": "@AppleSupport My iPhone 13 battery is dropping 80% to 10% in less than 2 hours."
    },
    {
        "label": "Billing & Refund Dispute",
        "text": "@AppleSupport I was double charged $14.99 for Apple Music this month. Need a refund!"
    },
    {
        "label": "Account / iCloud Lockout",
        "text": "@AppleSupport Forgot my Apple ID password and security questions. I'm locked out!"
    },
    {
        "label": "AirPods Connectivity",
        "text": "@AppleSupport My left AirPod Pro won't charge in the case or pair with my phone."
    },
    {
        "label": "Genius Bar Booking",
        "text": "@AppleSupport Can I book a Genius Bar appointment for screen repair at Regent Street?"
    }
]

@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form.get("message", "").strip() if request.method == "POST" else ""
    result = None
    if query:
        result = pipeline.process_message(query, evaluation_mode=True)
    return render_template("index.html", result=result, query=query, sample_queries=SAMPLE_QUERIES)

@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message parameter is required."}), 400
    
    result = pipeline.process_message(message, evaluation_mode=True)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Customer Support Automation Web UI on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
