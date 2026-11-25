
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw" /
    "dynamic_supply_chain_logistics_dataset.csv"
)


def load_shipments():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def investigate_shipment(row):
    incidents = []

    temperature = row["iot_temperature"]
    eta_delay = row["eta_variation_hours"]
    port_congestion = row["port_congestion_level"]
    risk = row["risk_classification"]
    delay_probability = row["delay_probability"]

    # SOP Rule 1: Temperature above 4°C
    if temperature > 4:
        incidents.append({
            "type": "Critical Temperature Breach",
            "severity": "Critical",
            "action": "Contact driver and restart auxiliary cooling unit.",
            "rule_reference": "SOP Section 1",
            "review_required": True,
            "applicability_note": (
                "Assumes fresh perishables; cargo type unavailable."
            )
        })

    # Below-zero readings: SOP does not prescribe a mitigation
    elif temperature < 0:
        incidents.append({
            "type": "Temperature Below SOP Range",
            "severity": "Review",
            "action": (
                "Flag for human review. SOP does not specify "
                "a mitigation for below-zero readings."
            ),
            "rule_reference": "SOP Section 1",
            "review_required": True,
            "applicability_note": (
                "Assumes fresh perishables; cargo type unavailable."
            )
        })

    # SOP Rule 1: Temperature breach plus ETA delay above 1 hour
    if temperature > 4 and eta_delay > 1:
        incidents.append({
            "type": "Emergency Cold Storage Diversion",
            "severity": "Critical",
            "action": (
                "Divert vehicle to the nearest emergency "
                "cold-storage facility."
            ),
            "rule_reference": "SOP Section 1",
            "review_required": True,
            "applicability_note": (
                "Temperature and ETA conditions met; "
                "human approval required."
            )
        })

    # SOP Rule 2: Port congestion above 7
    if port_congestion > 7:
        incidents.append({
            "type": "Port Congestion",
            "severity": "High",
            "action": (
                "Recommend diversion to Inland Empire "
                "Overflow Depot for cross-docking."
            ),
            "rule_reference": "SOP Section 2",
            "review_required": True,
            "applicability_note": (
                "Assumes Port of Long Beach / LA; "
                "port identity unavailable in dataset."
            )
        })

    # SOP Rule 3: High risk and delay probability above 0.65
    if risk == "High Risk" and delay_probability > 0.65:
        incidents.append({
            "type": "Tier 2 Escalation",
            "severity": "High",
            "action": "Escalate to Tier 2 Logistics Manager.",
            "rule_reference": "SOP Section 3",
            "review_required": True,
            "applicability_note": "SOP escalation condition satisfied."
        })

    return incidents


def get_incident_report(df):
    reports = []

    for index, row in df.iterrows():
        incidents = investigate_shipment(row)

        for incident in incidents:
            reports.append({
                "record_id": int(index),
                "timestamp": row["timestamp"],
                "latitude": row["vehicle_gps_latitude"],
                "longitude": row["vehicle_gps_longitude"],
                "temperature": row["iot_temperature"],
                "eta_delay": row["eta_variation_hours"],
                **incident
            })

    return pd.DataFrame(reports)


if __name__ == "__main__":
    shipments = load_shipments()
    report = get_incident_report(shipments)

    print("Total shipment records:", len(shipments))
    print("Total incident events:", len(report))
    print(
        "Unique affected records:",
        report["record_id"].nunique() if not report.empty else 0
    )

    if not report.empty:
        print("\nIncident breakdown:")
        print(report["type"].value_counts())

        print("\nLatest incidents:")
        print(
            report.sort_values("timestamp", ascending=False)
            .head(10)
            .to_string(index=False)
        )
