"""Disabled-by-default SMTP/SMS providers with persisted idempotent delivery."""
import hashlib
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from upgrade.notifications import eligible
from prediction.service import data_freshness
import config
from preprocessing.data_utils import file_sha256


def operational_approval(bundle):
    try:
        from experiments.finalize import verify_final
        verify_final()
        calibration = json.loads((config.RUN_DIR / 'calibration.json').read_text())
        return (calibration.get('operational_warning_validated') is True
                and bundle.get('model_sha256') == file_sha256(config.MODEL_PATH)
                and not bundle.get('sentinel_fusion', {}).get('used'))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def messages(bundle):
    results = []
    for row in bundle.get('daily_predictions', []):
        if row.get('warning_level') not in ('ELEVATED RISK', 'STRONG BLOOM RISK'):
            continue
        day = row['day']; date = bundle['forecast_dates'][day - 1]
        identifier = hashlib.sha256(f"{bundle['model_sha256']}:{date}:{row['warning_level']}".encode()).hexdigest()
        results.append({'id': identifier, 'region': 'Kerala project-supported coastal grid',
                        'risk_level': row['warning_level'], 'forecast_date': date, 'horizon_days': day,
                        'generated_at_utc': bundle.get('generated_at_utc'),
                        'data_quality': bundle['data_quality']['state'],
                        'source': bundle.get('satellite_source', 'MODIS')})
    return results


def send_email(message):
    required = ['HAB_SMTP_HOST', 'HAB_EMAIL_FROM', 'HAB_EMAIL_TO']
    if any(not os.getenv(k) for k in required):
        raise ValueError('SMTP provider is not configured.')
    mail = EmailMessage()
    mail['Subject'] = f"HAB proxy warning: {message['risk_level']} {message['forecast_date']}"
    mail['From'], mail['To'] = os.environ['HAB_EMAIL_FROM'], os.environ['HAB_EMAIL_TO']
    mail['Message-ID'] = f"<{message['id']}@hab.local>"
    mail.set_content(json.dumps(message, indent=2))
    with smtplib.SMTP(os.environ['HAB_SMTP_HOST'], int(os.getenv('HAB_SMTP_PORT', '587')), timeout=30) as client:
        client.starttls(context=ssl.create_default_context())
        if os.getenv('HAB_SMTP_USER'):
            client.login(os.environ['HAB_SMTP_USER'], os.environ['HAB_SMTP_PASSWORD'])
        if client.send_message(mail):
            raise ValueError('SMTP refused one or more recipients.')


def send_sms(message):
    # A configured gateway accepts this documented JSON contract; do not assume
    # arbitrary vendors accept it. Gateway must honor Idempotency-Key.
    endpoint, token, recipient = (os.getenv(k) for k in ('HAB_SMS_ENDPOINT', 'HAB_SMS_TOKEN', 'HAB_SMS_TO'))
    if not endpoint or urlparse(endpoint).scheme != 'https' or not token or not recipient:
        raise ValueError('HTTPS SMS gateway credentials/recipient are not configured.')
    request = Request(endpoint, data=json.dumps({'to': recipient, 'message': message}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token,
                 'Idempotency-Key': message['id']}, method='POST')
    with urlopen(request, timeout=30) as response:
        acknowledgement = json.loads(response.read())
    if acknowledgement.get('accepted') is not True or not acknowledgement.get('message_id'):
        raise ValueError('Gateway did not confirm acceptance.')
    return acknowledgement['message_id']


def dispatch(bundle, directory, dry_run=True):
    directory = Path(directory)
    # Dry runs can inspect research warnings, but must never preview unavailable,
    # low-coverage or stale predictions as deliverable operational alerts.
    available = bundle.get('status') == 'success' and bundle.get('data_quality', {}).get('state') == 'observed'
    fresh = bundle.get('freshness', {}).get('state') == 'fresh'
    if bundle.get('input_dates'):
        fresh = fresh and data_freshness(bundle['input_dates'][-1])['state'] == 'fresh'
    if not available or not fresh:
        return {'state': 'blocked', 'external_sent': False, 'entries': [], 'reason': 'Fresh sufficient observations required.'}
    payloads = messages(bundle)
    if dry_run:
        return {'state': 'dry_run', 'external_sent': False, 'operationally_eligible': eligible(bundle) and operational_approval(bundle), 'entries': payloads}
    if not eligible(bundle) or not operational_approval(bundle) or os.getenv('HAB_NOTIFICATIONS_ENABLED', '0') != '1':
        return {'state': 'disabled', 'external_sent': False, 'entries': []}
    provider = os.getenv('HAB_NOTIFICATION_PROVIDER', '')
    if provider not in ('email', 'sms'):
        return {'state': 'not_configured', 'external_sent': False, 'entries': []}
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for payload in payloads:
        path = directory / f"{provider}_{payload['id']}.json"
        try:
            # Claim before send: uncertain/crashed sends require manual inspection;
            # never blindly retry SMTP, which has no idempotency guarantee.
            with path.open('x') as handle:
                json.dump({'state': 'pending', 'payload': payload}, handle)
        except FileExistsError:
            entries.append({'id': payload['id'], 'state': 'already_claimed'})
            continue
        try:
            receipt = send_sms(payload) if provider == 'sms' else send_email(payload)
            result = {'id': payload['id'], 'state': 'provider_accepted', 'provider': provider,
                      'external_sent': True, 'receipt': receipt, 'processed_at_utc': datetime.now(timezone.utc).isoformat()}
        except Exception as error:
            result = {'id': payload['id'], 'state': 'delivery_failed_or_uncertain', 'external_sent': False,
                      'error_type': type(error).__name__}
        path.write_text(json.dumps({'payload': payload, **result}, indent=2))
        entries.append(result)
    return {'state': 'processed', 'external_sent': any(e.get('external_sent') for e in entries), 'entries': entries}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--outbox', type=Path, default=Path('outputs/live/delivery'))
    parser.add_argument('--deliver', action='store_true', help='Requires explicit enable configuration AND validated warning gate. Default is dry-run.')
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    print(json.dumps(dispatch(bundle, args.outbox, dry_run=not args.deliver), indent=2))


if __name__ == '__main__':
    main()
