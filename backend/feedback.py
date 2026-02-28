from flask import Blueprint, request, jsonify
import csv
import os
from datetime import datetime

feedback_bp = Blueprint("feedback", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEEDBACK_FILE = os.path.join(BASE_DIR, "data", "feedback.csv")


@feedback_bp.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    try:
        # Get form data from the feedback form
        accuracy = request.form.get("accuracy")
        comments = request.form.get("comments", "").strip()
        prediction = request.form.get("prediction", "Unknown")

        # Validate that accuracy rating is provided
        if not accuracy:
            return jsonify({"status": "error", "message": "Please select an accuracy rating"}), 400

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction,
            accuracy,
            comments
        ]

        file_exists = os.path.isfile(FEEDBACK_FILE)

        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["timestamp", "prediction", "accuracy", "comment"])

            writer.writerow(row)

        return jsonify({"status": "success", "message": "Thank you for your feedback!"})
    
    except Exception as e:
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to submit feedback"}), 500