"""Read verified research forecast bundles for exports and observation context."""
import csv
import io
import json
import numpy as np
import config
from preprocessing.data_utils import file_sha256, environment_fingerprint
from prediction.service import decorate_bundle, validate_current_contract


def load_bundle(identifier, demo=False):
    if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Invalid forecast identifier.')
    directory = config.FORECAST_DIR / identifier
    bundle = json.loads((directory / 'bundle.json').read_text())
    training = json.loads((config.RUN_DIR / 'training_config.json').read_text())
    validate_current_contract(training)
    if (bundle.get('forecast_id') != identifier or training.get('smoke') or not training.get('epochs_completed')
            or training.get('model_sha256') != bundle.get('model_sha256')
            or bundle.get('model_sha256') != file_sha256(config.MODEL_PATH)
            or bundle.get('preprocessing_sha256') != file_sha256(config.SEQUENCE_DIR / 'preprocessing.json')
            or bundle.get('calibration_sha256') != file_sha256(config.RUN_DIR / 'calibration.json')):
        raise ValueError('Forecast differs from the active model/preprocessing/decision artifacts.')
    manifest = json.loads((config.SEQUENCE_DIR / 'manifest.json').read_text())
    if manifest['environment_fingerprint'] != environment_fingerprint(config.PROCESSED_DIR):
        raise ValueError('Environmental inputs changed after fitting; forecast context is unavailable.')
    if bundle.get('status') != 'success':
        raise ValueError('This bundle has no available forecast.')
    for name in ('prediction.npy', 'risk_classes.npy', 'support.npy'):
        if bundle.get('array_sha256', {}).get(name) != file_sha256(directory / name):
            raise ValueError('Forecast array integrity check failed.')
    decorated = decorate_bundle(bundle, allow_demo=demo)
    if decorated.get('status') != 'success':
        raise ValueError('Historical data require explicit demonstration mode.')
    return decorated


def summary_csv(bundle):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['forecast_id', 'input_end_date', 'forecast_date', 'day', 'warning_level',
                     'strong_class_percent', 'strong_mean_score', 'input_coverage_percent',
                     'freshness', 'operational_warning_available', 'label_semantics'])
    for row, date in zip(bundle['daily_predictions'], bundle['forecast_dates'], strict=True):
        writer.writerow([bundle['forecast_id'], bundle['input_dates'][-1], date, row['day'], row['warning_level'],
                         row['high_risk_percentage'], row.get('mean_probability'),
                         bundle['data_quality']['chlorophyll_mean_input_coverage_percent'], bundle['freshness']['state'],
                         bundle.get('operational_warnings_available', False), 'chlorophyll anomaly proxy'])
    return output.getvalue()


def observation_map(bundle):
    from preprocessing.pipeline import EnvironmentalPipeline
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    pipeline = EnvironmentalPipeline()
    found = np.flatnonzero(pipeline.dates == np.datetime64(bundle['input_dates'][-1]))
    if not len(found) or found[0] < 4:
        raise ValueError('Archived observation context is unavailable for this input date.')
    end = int(found[0]); indices = np.arange(end-4, end+1)
    _, valid = pipeline.valid('chlorophyll', indices)
    latest = pipeline.labels[end]
    fig = Figure(figsize=(10, 5), layout='constrained')
    axes = fig.subplots(1, 2)
    cmap = ListedColormap(['#4caf50', '#f2c94c', '#e74c3c']).with_extremes(bad='#e5e7eb')
    left = axes[0].pcolormesh(pipeline.longitude, pipeline.latitude, np.ma.masked_where(latest == 255, latest),
                             cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5], 3), shading='nearest')
    fig.colorbar(left, ax=axes[0], ticks=[0,1,2], shrink=.75).ax.set_yticklabels(['Normal','Elevated','Strong'])
    right = axes[1].pcolormesh(pipeline.longitude, pipeline.latitude, np.ma.masked_where(~pipeline.ocean, valid.sum(0)),
                              vmin=0, vmax=5, cmap='Blues', shading='nearest')
    fig.colorbar(right, ax=axes[1], ticks=range(6), shrink=.75, label='Observed input days')
    axes[0].set_title(f'Observed proxy labels · {bundle["input_dates"][-1]}')
    axes[1].set_title('Chlorophyll observations in five input days')
    for ax in axes:
        ax.set(xlabel='Longitude (°E)', ylabel='Latitude (°N)', facecolor='#e5e7eb')
    fig.suptitle('Historical observation context · grey = unobserved or unsupported\nProxy labels do not confirm harmful species or toxicity')
    image = io.BytesIO(); fig.savefig(image, format='png', dpi=120)
    return image.getvalue()
