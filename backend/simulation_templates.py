from __future__ import annotations

# Each template only fills TransactionIn's scoring-relevant input fields.
# isFraud is never included — whether a transaction is fraud is for the
# model + rule engine to decide, not the template.
SIMULATION_TEMPLATES: list[dict] = [
    {
        "id": "account_drain",
        "label": "Account Drain",
        "description": (
            "Origin account fully drained to a destination with no balance "
            "history — the pattern the model has learned to flag most strongly."
        ),
        "input": {
            "step": 3,
            "type": "TRANSFER",
            "amount": 320000.0,
            "oldbalanceOrg": 320000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
    },
    {
        "id": "normal_transfer",
        "label": "Normal Transfer",
        "description": (
            "Modest daytime transfer typical of PaySim's clean-transaction "
            "majority (origin balance untracked/zero, as most non-fraud rows "
            "are generated) to a destination with an existing balance."
        ),
        "input": {
            "step": 14,
            "type": "TRANSFER",
            "amount": 5000.0,
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 18340.0,
            "newbalanceDest": 23340.0,
        },
    },
    {
        "id": "borderline",
        "label": "Borderline",
        "description": (
            "A transfer over the high-amount threshold (>200k) but a partial, "
            "balance-consistent drain to a real destination — not an automatic "
            "hard-rule hit."
        ),
        "input": {
            "step": 16,
            "type": "TRANSFER",
            "amount": 250000.0,
            "oldbalanceOrg": 900000.0,
            "newbalanceOrig": 650000.0,
            "oldbalanceDest": 12000.0,
            "newbalanceDest": 262000.0,
        },
    },
    {
        "id": "high_amount_cashout",
        "label": "High-Amount Cash-Out",
        "description": (
            "A large cash-out with a near-complete drain to a merchant-style "
            "destination (zero balance before/after) — typically trips the "
            "ghost_destination hard rule (a known PaySim artifact: merchant "
            "accounts never update their balance)."
        ),
        "input": {
            "step": 20,
            "type": "CASH_OUT",
            "amount": 495000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
    },
]
