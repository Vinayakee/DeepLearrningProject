"""Additional source manifest validation shared by live and hindcast entrypoints."""
from datetime import datetime, timezone
import numpy as np
import config


def validate_manifest(directory):
    import json
    from pathlib import Path
    metadata = json.loads((Path(directory) / 'metadata.json').read_text())
    sources = metadata.get('sources')
    if not isinstance(sources, dict) or set(sources) != set(config.CHANNEL_NAMES):
        raise ValueError('Require provenance for exactly the seven environmental channels.')
    for source in sources.values():
        if not isinstance(source, dict) or not isinstance(source.get('product_ids'), list) or not source['product_ids'] or any(not isinstance(p, str) or not p.strip() for p in source['product_ids']):
            raise ValueError('Nonempty source product IDs are required.')
        times = []
        for key in ('observation_end_utc', 'available_at_utc'):
            stamp = source.get(key)
            if not isinstance(stamp, str) or not stamp.endswith('Z'):
                raise ValueError('Actual observation/publication timestamps must end Z.')
            time = datetime.fromisoformat(stamp[:-1] + '+00:00')
            if time > datetime.now(timezone.utc):
                raise ValueError('Source acquisition/publication cannot be in the future.')
            times.append(time)
        if times[1] < times[0]:
            raise ValueError('Source availability cannot precede acquisition end.')
    with np.load(Path(directory) / 'inputs.npz', allow_pickle=False) as inputs:
        dates = inputs['dates'].astype('datetime64[D]')
        for i, name in enumerate(config.CHANNEL_NAMES):
            observed_days = np.flatnonzero(inputs['masks'][..., i].sum((1, 2)))
            if len(observed_days):
                end = np.datetime64(sources[name]['observation_end_utc'][:-1], 's').astype('datetime64[D]')
                if end < dates[observed_days[-1]]:
                    raise ValueError('Source acquisition end precedes a claimed observed frame.')
    return metadata
