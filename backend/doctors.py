from flask import Blueprint, render_template, request
import csv

doctors_bp = Blueprint("doctors", __name__)

DATA_PATH = "data/dermatologists.csv"


def load_doctors():
    with open(DATA_PATH, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f))


@doctors_bp.route("/doctors")
def doctors():

    city_filter = request.args.get("city")

    doctors = load_doctors()

    # Filter by city only
    if city_filter and city_filter != "All":
        doctors = [
            d for d in doctors
            if d["city"].strip().lower() == city_filter.strip().lower()
        ]

    # build city list
    all_docs = load_doctors()
    cities = sorted(set(d["city"] for d in all_docs))

    return render_template(
        "doctors.html",
        doctors=doctors,
        cities=cities,
        selected_city=city_filter
    )