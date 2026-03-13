from flask import Blueprint, render_template, session, redirect, url_for, flash, request
# from models import PredictionHistory
from models import db,PredictionHistory, User
from auth_utils import login_required
import os

profile_bp = Blueprint("profile", __name__)

@profile_bp.route("/profile")
@login_required
def profile():
    user_id = session.get("user_id")
    
    # Fetch User details
    user = User.query.get(user_id)

    # Fetch user's prediction history sorted by most recent
    history = PredictionHistory.query.filter_by(user_id=user_id).order_by(PredictionHistory.timestamp.desc()).all()
    
    # return render_template("profile.html", history=history)
    return render_template("profile.html", user=user, history=history)

@profile_bp.route("/profile/clear", methods=["POST"])
@login_required
def clear_history():
    user_id = session.get("user_id")
    
    PredictionHistory.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    
    flash("Prediction history cleared successfully.")
    return redirect(url_for("profile.profile"))

@profile_bp.route("/profile/update", methods=["POST"])
@login_required
def update_profile():
    user_id = session.get("user_id")
    user = User.query.get(user_id)
    
    new_name = request.form.get("name")
    new_email = request.form.get("email")
    
    if new_name and new_email:
        # Check if email exists for another user
        existing_user = User.query.filter_by(email=new_email).first()
        if existing_user and existing_user.id != user_id:
            flash("Email is already registered by another account.", "danger")
        else:
            user.name = new_name
            user.email = new_email
            db.session.commit()
            session["user_name"] = new_name  # Update session
            flash("Profile updated successfully!", "success")
    else:
        flash("Please provide both name and email.", "danger")
        
    return redirect(url_for("profile.profile"))