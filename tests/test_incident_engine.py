
import pandas as pd
from src.incident_engine import investigate_shipment


def make_shipment(**overrides):
    data = {
        "iot_temperature": 2.0,
        "eta_variation_hours": 0.5,
        "port_congestion_level": 3.0,
        "risk_classification": "Low Risk",
        "delay_probability": 0.2,
    }
    data.update(overrides)
    return pd.Series(data)


def test_critical_temperature_breach():
    incidents = investigate_shipment(
        make_shipment(iot_temperature=5.0)
    )
    assert any(i["type"] == "Critical Temperature Breach" for i in incidents)


def test_below_zero_requires_review():
    incidents = investigate_shipment(
        make_shipment(iot_temperature=-2.0)
    )
    incident = next(i for i in incidents if i["type"] == "Temperature Below SOP Range")
    assert incident["severity"] == "Review"
    assert "human review" in incident["action"]


def test_emergency_cold_storage():
    incidents = investigate_shipment(
        make_shipment(iot_temperature=5.0, eta_variation_hours=2.0)
    )
    assert any(i["type"] == "Emergency Cold Storage Diversion" for i in incidents)


def test_port_congestion():
    incidents = investigate_shipment(
        make_shipment(port_congestion_level=8.0)
    )
    assert any(i["type"] == "Port Congestion" for i in incidents)


def test_tier_two_escalation():
    incidents = investigate_shipment(
        make_shipment(
            risk_classification="High Risk",
            delay_probability=0.8
        )
    )
    assert any(i["type"] == "Tier 2 Escalation" for i in incidents)


def test_no_incidents():
    incidents = investigate_shipment(make_shipment())
    assert incidents == []


def test_temperature_boundary():
    incidents = investigate_shipment(make_shipment(iot_temperature=4.0))
    assert not any(
        i["type"] == "Critical Temperature Breach"
        for i in incidents
    )
