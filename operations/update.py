"""One-shot updates and an optional configurable polling worker."""
import argparse
import json
import os
import time
import uuid
import socket
from pathlib import Path
from datetime import datetime, timezone
from upgrade import live
from preprocessing.pipeline import EnvironmentalPipeline
from operations.acquisition import acquire_with_retry, AcquisitionUnavailable
from operations.fusion import stage_fusion
from operations.notifications import dispatch
from operations.provenance import validate_manifest
from preprocessing.data_utils import file_sha256


def recover_lock():
    """Explicit recovery only when the recorded same-host owner no longer exists."""
    lock = live.ROOT / 'update.lock'
    record = json.loads(lock.read_text())
    if record.get('hostname') != socket.gethostname() or not isinstance(record.get('pid'), int) or record['pid'] <= 0:
        raise ValueError('Cannot prove lock ownership; inspect the lock manually.')
    try:
        os.kill(record['pid'], 0)
    except ProcessLookupError:
        if json.loads(lock.read_text()) != record:
            raise ValueError('Lock owner changed during recovery.')
        lock.unlink()
        return {'state': 'orphan_lock_recovered', 'live': False}
    except PermissionError as error:
        raise ValueError('Cannot prove the lock owner is absent.') from error
    raise ValueError('Lock owner is still running; recovery refused.')


def run_update(directory=None, sentinel_dir=None, research_fusion=False):
    # Cross-process lock also protects against overlapping scheduled/manual runs.
    live.ROOT.mkdir(parents=True, exist_ok=True)
    lock = live.ROOT / 'update.lock'
    try:
        handle = lock.open('x')
    except FileExistsError:
        return {'state': 'update_in_progress', 'live': False, 'reason': 'Another worker owns update.lock.'}
    try:
        token = uuid.uuid4().hex
        with handle:
            handle.write(json.dumps({'pid': os.getpid(), 'hostname': socket.gethostname(), 'token': token,
                                     'started_at_utc': datetime.now(timezone.utc).isoformat()}))
        try:
            acquired = acquire_with_retry(directory, attempts=int(os.getenv('HAB_ACQUISITION_ATTEMPTS', '3')))
            validate_manifest(acquired)
            source = {'state': 'disabled', 'used': False, 'source': 'MODIS'}
            if research_fusion:
                acquired, source = stage_fusion(acquired, sentinel_dir or os.getenv('HAB_SENTINEL_DIR'),
                    live.ROOT / 'staging' / uuid.uuid4().hex, EnvironmentalPipeline())
            result = live.update(acquired)
            result['sentinel_fusion'] = source
            quality = result.get('data_quality', {}).get('state')
            if result['state'] == 'insufficient_data' and quality in ('missing_data', 'low_coverage'):
                result['state'] = quality
            if result['state'] == 'fresh_research_prediction':
                path = live.ROOT / 'bundles' / result['latest_successful_bundle'] / 'bundle.json'
                bundle = json.loads(path.read_text())
                bundle.update(satellite_source=source['source'], sentinel_fusion=source,
                    calibrated_probability_available=False, model_version='frozen_final')
                # Unvalidated fusion can never enable production warnings.
                if source['used']:
                    bundle.update(operational_warning_validated=False, operational_warnings_available=False,
                                  warning_reason='Experimental Sentinel fusion; operational warnings disabled.')
                    provenance = acquired / 'chlorophyll_provenance.npy'
                    target = path.parent / provenance.name
                    target.write_bytes(provenance.read_bytes())
                    bundle['chlorophyll_provenance_sha256'] = file_sha256(target)
                path.write_text(json.dumps(bundle, indent=2))
                result['notifications'] = dispatch(bundle, live.ROOT / 'delivery', dry_run=os.getenv('HAB_NOTIFICATION_DRY_RUN', '1') != '0')
            live.write_status(live.ROOT / 'status.json', result)
            return result
        except Exception as error:
            previous = live.current_status()
            result = {'state': 'not_configured' if isinstance(error, AcquisitionUnavailable) and not directory and not os.getenv('HAB_LIVE_INPUT_DIR') and not os.getenv('HAB_LIVE_ADAPTER') else 'acquisition_failed',
                      'live': False, 'last_attempt_utc': datetime.now(timezone.utc).isoformat(),
                      'last_successful_update_utc': previous.get('last_successful_update_utc'),
                      'latest_successful_bundle': previous.get('latest_successful_bundle'),
                      'reason': f'Real acquisition/validation unavailable ({type(error).__name__}); no prediction claimed.'}
            live.write_status(live.ROOT / 'status.json', result)
            return result
    finally:
        try:
            if json.loads(lock.read_text()).get('token') == token:
                lock.unlink()
        except (OSError, ValueError):
            pass  # A changed/unreadable lock must be inspected, never removed blindly.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--sentinel-dir', type=Path)
    parser.add_argument('--research-fusion', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Explicitly run a polling worker; acquisition disabled without configuration.')
    parser.add_argument('--recover-lock', action='store_true', help='Recover only a provably abandoned same-host worker lock, then exit.')
    parser.add_argument('--interval-seconds', type=int, default=int(os.getenv('HAB_UPDATE_INTERVAL_SECONDS', '3600')))
    args = parser.parse_args()
    if args.interval_seconds < 60:
        parser.error('Polling interval must be at least 60 seconds.')
    if args.recover_lock:
        try:
            print(json.dumps(recover_lock()))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    try:
        while True:
            print(json.dumps(run_update(args.input_dir, args.sentinel_dir, args.research_fusion), indent=2), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
