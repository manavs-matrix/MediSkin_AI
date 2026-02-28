from flask import Flask, session, render_template
from config import Config
from models import db

from auth import auth_bp
from predict import predict_bp
from doctors import doctors_bp
from auth_utils import login_required
from feedback import feedback_bp

# -----------------------
# Create App FIRST
# -----------------------
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = "mediskin-secret-key"

# -----------------------
# Init DB
# -----------------------
db.init_app(app)

# -----------------------
# Register Blueprints
# -----------------------
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(doctors_bp)
app.register_blueprint(feedback_bp)

# -----------------------
# Protected Home Route
# -----------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")


# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)

