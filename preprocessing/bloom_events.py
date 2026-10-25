"""Training-defined satellite bloom-event labels; never confirmed harmful blooms."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label, generate_binary_structure
import config
from preprocessing.pipeline import EnvironmentalPipeline, build_sequence_index
from preprocessing.data_utils import file_sha256, month_numbers, environment_fingerprint

TITLE = 'Satellite bloom-event forecast'
DEFAULT_DATASET = config.ROOT/'data/processed/bloom_events/dataset_01'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def patch_filter(mask, minimum):
    if minimum < 1:
        raise ValueError('Patch size must be positive.')
    components, _ = label(mask)  # four-neighbour spatial connectivity
    counts = np.bincount(components.ravel())
    keep = counts >= minimum
    keep[0] = False
    return keep[components]


def confirmed_membership(high, minimum):
    """A fixed common patch must be high on all three observed consecutive days.

    Retrospective membership marks all three days. As-of flags mark only the
    third day, so the persistence comparator never reads future confirmation.
    """
    high = np.asarray(high, bool)
    if high.ndim != 3:
        raise ValueError('Expected time/latitude/longitude high flags.')
    members, asof = np.zeros_like(high), np.zeros_like(high)
    for start in range(len(high)-2):
        patch = patch_filter(high[start:start+3].all(0), minimum)
        members[start:start+3] |= patch
        asof[start+2] |= patch
    return members, asof


def event_labels(high, observed, support, minimum):
    """Do not convert missing days, uncertain high runs or split edges to negatives."""
    high, observed, support = np.asarray(high, bool), np.asarray(observed, bool), np.asarray(support, bool)
    if high.shape != observed.shape or high.ndim != 3 or high.shape[1:] != support.shape:
        raise ValueError('High flags, observations and spatial support differ.')
    if np.any(high & ~observed) or np.any(observed & ~support):
        raise ValueError('High flags require actual observations inside support.')
    members, confirmed = confirmed_membership(high, minimum)
    # Missing observations might have been high. Evaluate this upper bound only
    # to withhold uncertain negatives; it never creates positive target labels.
    possible = high | (~observed & support)
    padded = np.concatenate([np.broadcast_to(support, (2,*support.shape)), possible,
                             np.broadcast_to(support, (2,*support.shape))])
    possible_members, _ = confirmed_membership(padded, minimum)
    ambiguous = observed & high & possible_members[2:-2] & ~members
    output = np.full(high.shape, 255, np.uint8)
    output[observed & ~ambiguous] = 0
    output[members & observed] = 1
    # At issue time only measured non-high pixels and already confirmed runs
    # have a known event state. A new high run might still develop into an event.
    asof = np.full(high.shape, 255, np.uint8)
    asof[observed & ~high] = 0
    asof[confirmed & observed] = 1
    return output, asof


def event_summary(labels):
    components, count = label(labels == 1, structure=generate_binary_structure(3,1))
    observed = labels != 255
    sizes = np.bincount(components.ravel())[1:]
    return {'events': int(count), 'positive_pixel_days': int((labels == 1).sum()),
            'negative_pixel_days': int((labels == 0).sum()), 'unknown_pixel_days': int((labels == 255).sum()),
            'valid_pixel_days': int(observed.sum()),
            'positive_fraction_of_valid': float((labels == 1).sum()/observed.sum()) if observed.any() else None,
            'largest_event_pixel_days': int(sizes.max()) if len(sizes) else 0,
            'few_events_warning': bool(count < 10)}


def prepare(output=DEFAULT_DATASET):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh satellite bloom-event dataset directory.')
    p = EnvironmentalPipeline()
    if (p.stats['train_start'],p.stats['train_end']) != ('2021-01-01','2023-12-31'):
        raise ValueError('Only 2021–2023 fitting artifacts may be reused.')
    label_meta = p.metadata
    if (label_meta['train_start'], label_meta['train_end']) != ('2021-01-01','2023-12-31'):
        raise ValueError('The anomaly thresholds were not fitted on the training period.')
    anomaly = np.load(p.label_dir/'chlorophyll_anomaly_2021_2025.npy', mmap_mode='r')
    thresholds = np.load(p.label_dir/'hab_p90_monthly.npy')
    months = month_numbers(p.dates)
    observed = np.isfinite(anomaly) & p.ocean
    high = observed & (anomaly > thresholds[months-1,None,None])
    train = (p.dates >= np.datetime64('2021-01-01')) & (p.dates <= np.datetime64('2023-12-31'))
    candidates = []
    for minimum in (1,2,3,5,10,20):
        members,_ = confirmed_membership(high[train], minimum)
        count = label(members, structure=generate_binary_structure(3,1))[1]
        candidates.append({'minimum_patch_pixels': minimum, 'training_events': int(count),
                           'training_positive_pixel_days': int(members.sum())})
    # A compact non-singleton patch while retaining an analysable training set.
    # This criterion is applied only to the training counts, never held-out data.
    singleton_positive = candidates[0]['training_positive_pixel_days']
    eligible = [r for r in candidates if r['minimum_patch_pixels'] >= 2 and r['training_events'] >= 20
                and r['training_positive_pixel_days'] >= .8*singleton_positive]
    if not eligible:
        raise ValueError('No non-singleton patch meets the declared training-support criterion; do not fabricate event labels.')
    chosen = max(eligible, key=lambda r:r['minimum_patch_pixels'])['minimum_patch_pixels']
    labels, asof = np.full(high.shape,255,np.uint8),np.full(high.shape,255,np.uint8)
    reports = {}
    for split,(first,last) in config.SPLITS.items():
        selected = (p.dates >= np.datetime64(first)) & (p.dates <= np.datetime64(last))
        labels[selected],asof[selected] = event_labels(high[selected],observed[selected],p.ocean,chosen)
        reports[split] = {**event_summary(labels[selected]), 'period':[first,last],
                          'observed_chlorophyll_pixel_days': int(observed[selected].sum()),
                          'uncertain_observed_pixel_days': int((observed[selected] & (labels[selected] == 255)).sum())}
    index = build_sequence_index(p.dates,labels)
    for split in config.SPLITS:
        reports[split]['windows'] = int((index.split == split).sum())
    output.mkdir(parents=True)
    np.save(output/'event_labels.npy',labels)
    np.save(output/'confirmed_asof.npy',asof)
    np.save(output/'support.npy',p.ocean)
    index.to_csv(output/'sequence_index.csv',index=False)
    source_paths = [p.label_dir/name for name in ('label_metadata.json','climatology_metadata.json',
                    'monthly_chl_climatology_2021_2025.npy','hab_p90_monthly.npy')]+[p.sequence_dir/'preprocessing.json']
    metadata = {'schema':'satellite-bloom-events-v1','title':TITLE,'minimum_patch_pixels':chosen,
                'duration_days':3,'spatial_connectivity':4,'event_connectivity':'6-neighbour observed space/time components',
                'definition':'Strictly above training monthly relative-anomaly P90; a common four-connected patch of at least M pixels is observed high on every one of three consecutive days. All three days are event members.',
                'unknown_policy':'Unobserved/unsupported, potentially qualifying high runs interrupted by missing observations, and unresolved split-edge high runs remain 255. Unknown never becomes a positive or a negative.',
                'retrospective_confirmation':'Membership may require two following days; as-of persistence uses only completed trailing three-day confirmations. No triples cross splits.',
                'patch_selection':{'training_candidates':candidates,'criterion':'Largest M>=2 retaining at least 20 training components and 80% of singleton positive pixel-days; selected without validation/test counts.'},
                'fit_period':['2021-01-01','2023-12-31'],'monthly_p90':thresholds.tolist(),
                'base_label_dir':str(p.label_dir),'base_sequence_dir':str(p.sequence_dir),
                'fitted_sources':{str(path):file_sha256(path) for path in source_paths},'split_reports':reports,
                'limitations':'Chlorophyll-derived satellite event membership, not verified harmful species, toxicity or field HAB occurrence. Components split by observation gaps are not independent biological events.',
                'test_disclosure':'2025 was previously inspected for the original project. New model and threshold are frozen before one new satellite-event test evaluation.'}
    write_json(output/'label_metadata.json',metadata)
    stats = deepcopy(p.stats)
    stats.update(seasonal_features=True,label_metadata_sha256=file_sha256(output/'label_metadata.json'))
    write_json(output/'preprocessing.json',stats)
    files = ('event_labels.npy','confirmed_asof.npy','support.npy','sequence_index.csv','label_metadata.json','preprocessing.json')
    manifest = {'schema':'satellite-event-dataset-v1','label_metadata_sha256':file_sha256(output/'label_metadata.json'),
                'sequence_index_sha256':file_sha256(output/'sequence_index.csv'),'environment_fingerprint':p.environment_identity,
                'files':{name:file_sha256(output/name) for name in files}}
    write_json(output/'manifest.json',manifest)
    print(json.dumps({'selected_M':chosen,'training_candidates':candidates,'splits':reports},indent=2),flush=True)
    return metadata


class BloomPipeline(EnvironmentalPipeline):
    def __init__(self,dataset=DEFAULT_DATASET):
        dataset = Path(dataset)
        manifest = json.loads((dataset/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if file_sha256(dataset/name) != digest:
                raise ValueError('Satellite-event dataset changed: '+name)
        metadata = json.loads((dataset/'label_metadata.json').read_text())
        for name,digest in metadata['fitted_sources'].items():
            if file_sha256(name) != digest:
                raise ValueError('Training-fitted source changed: '+name)
        super().__init__(label_dir=metadata['base_label_dir'],sequence_dir=metadata['base_sequence_dir'])
        if self.environment_identity != manifest['environment_fingerprint']:
            raise ValueError('Environmental archive changed since satellite-event preparation.')
        self.labels = np.load(dataset/'event_labels.npy',mmap_mode='r')
        self.asof = np.load(dataset/'confirmed_asof.npy',mmap_mode='r')
        self.label_dir = self.sequence_dir = dataset
        self.stats = json.loads((dataset/'preprocessing.json').read_text())
        self.event_metadata = metadata
        if self.labels.shape != self.shape or not np.isin(self.labels,[0,1,255]).all():
            raise ValueError('Invalid satellite-event targets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_DATASET)
    args=parser.parse_args();prepare(args.output_dir)


if __name__=='__main__':main()
