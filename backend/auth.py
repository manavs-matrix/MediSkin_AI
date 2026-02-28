from flask import Blueprint, request, redirect, url_for, render_template, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User

auth_bp = Blueprint("auth", __name__)


# =========================
# REGISTER
# =========================
@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if not name or not email or not password:
            flash("All fields are required")
            return redirect(url_for("auth.register"))

        if password != confirm:
            flash("Passwords do not match")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered")
            return redirect(url_for("auth.register"))

        hashed_pw = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password_hash=hashed_pw
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful — please login")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


# =========================
# LOGIN
# =========================
@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("No account with this email")
            return redirect(url_for("auth.login"))

        if not check_password_hash(user.password_hash, password):
            flash("Incorrect password")
            return redirect(url_for("auth.login"))

        session["user_id"] = user.id
        session["user_name"] = user.name

        flash("Login successful")

        # ✅ FIXED REDIRECT
        return redirect(url_for("predict.predict"))

    return render_template("login.html")


# =========================
# LOGOUT
# =========================
@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect(url_for("auth.login"))

