"""Truthful deployment metadata shared by health and system endpoints."""
import json
import os
import config


def notification_status():
    try:
        approved = json.loads((config.RUN_DIR / 'calibration.json').read_text()).get('operational_warning_validated') is True
    except (OSError, ValueError, TypeError):
        approved = False
    return {'external_delivery': False, 'mode': 'disabled' if not approved else 'provider configuration required',
            'providers_configured': {'smtp': all(os.getenv(k) for k in ('HAB_SMTP_HOST', 'HAB_EMAIL_FROM', 'HAB_EMAIL_TO')),
                                     'sms': all(os.getenv(k) for k in ('HAB_SMS_ENDPOINT', 'HAB_SMS_TOKEN', 'HAB_SMS_TO'))},
            'operational_warning_validated': approved,
            'reason': 'Regional warning reliability has not been validated.' if not approved else 'Provider enablement is reported by individual delivery attempts.'}


def calibration_status():
    path = config.ROOT / 'outputs/upgrade/calibration_2024.json'
    try:
        experiment = json.loads(path.read_text())
        tested = bool(experiment.get('temperature'))
    except (OSError, ValueError, TypeError):
        tested = False
    return {'artifact_exists': path.exists(), 'tested': tested,
            'decision_rule_artifact_exists': (config.RUN_DIR / 'calibration.json').exists(),
            'active': False, 'operationally_approved': False,
            'interpretation': 'Raw model scores. Temperature scaling was assessed on 2024 development data only; it is not deployed.'}
