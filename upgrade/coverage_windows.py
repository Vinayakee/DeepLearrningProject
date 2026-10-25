"""List supported development-date input windows without running the model."""
import numpy as np
from prediction.coverage import DEFAULT_POLICY


def covered_windows(pipeline,year=2024):
    if year!=2024:
        raise ValueError('This development-date helper lists 2024 only; no holdout selection.')
    counts=[]
    # Observation masks only; target labels and model scores do not select these dates.
    selected=np.flatnonzero((pipeline.dates>=np.datetime64('2024-01-01'))&(pipeline.dates<=np.datetime64('2024-12-31')))
    if not len(selected):return []
    first=max(int(selected[0])-4,0);last=int(selected[-1]);indices=np.arange(first,last+1)
    for name in pipeline.stats['channel_order']:
        _,valid=pipeline.valid(name,indices);counts.append(valid.sum((1,2)))
    counts=np.stack(counts,-1);marine=int(pipeline.ocean.sum());windows=[]
    for index in selected:
        position=int(index)-first
        if position<4:continue
        sample=counts[position-4:position+1]
        observed_days=(sample>0).sum(0)
        coverage=float(sample[:,0].sum()/(5*marine)*100)
        if coverage>=DEFAULT_POLICY['minimum_chlorophyll_mean_percent'] and observed_days[0]>=DEFAULT_POLICY['minimum_chlorophyll_observed_days'] and np.all(observed_days[1:]>=DEFAULT_POLICY['minimum_environment_observed_days']):
            windows.append({'input_end_date':str(pipeline.dates[index]),'mean_chlorophyll_coverage_percent':coverage,'chlorophyll_observed_days':int(observed_days[0])})
    return windows
