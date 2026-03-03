from flask import Blueprint, render_template, request
import csv

hospitals_bp = Blueprint("hospitals", __name__)

DATA_PATH = "data/hospitals.csv"


def load_hospitals():
    with open(DATA_PATH, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f))


@hospitals_bp.route("/hospitals")
def hospitals():

    state_filter = request.args.get("state")
    district_filter = request.args.get("district")

    hospitals_list = load_hospitals()

    # Filter by state and district
    if state_filter and state_filter != "All":
        hospitals_list = [
            h for h in hospitals_list
            if h["state"].strip().lower() == state_filter.strip().lower()
        ]
    
    if district_filter and district_filter != "All":
        hospitals_list = [
            h for h in hospitals_list
            if h["district"].strip().lower() == district_filter.strip().lower()
        ]

    # build state and district lists
    all_hospitals = load_hospitals()
    states = sorted(set(h["state"] for h in all_hospitals))
    
    # Get districts based on selected state
    if state_filter and state_filter != "All":
        districts = sorted(set(
            h["district"] for h in all_hospitals 
            if h["state"].strip().lower() == state_filter.strip().lower()
        ))
    else:
        districts = sorted(set(h["district"] for h in all_hospitals))

    return render_template(
        "hospitals.html",
        hospitals=hospitals_list,
        states=states,
        districts=districts,
        selected_state=state_filter,
        selected_district=district_filter
    )
