from functools import wraps
from flask import session, redirect, url_for, flash


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first")
            return redirect(url_for("auth.login"))

        return view_func(*args, **kwargs)

    return wrapper
