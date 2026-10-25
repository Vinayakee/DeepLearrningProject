"""Read one real NASA L3 mapped chlorophyll file without rewriting the archive."""
from pathlib import Path
import numpy as np


def read_daily(path,target_latitude,target_longitude):
    import xarray as xr
    path=Path(path)
    with xr.open_dataset(path,mask_and_scale=True) as dataset:
        variable=dataset['chlor_a']
        if variable.attrs.get('units') not in ('mg m^-3','mg m-3','mg/m3'):
            raise ValueError('Expected decoded linear chlorophyll concentration units.')
        if variable.dims!=('lat','lon'):
            raise ValueError('Require a daily two-dimensional mapped L3 product.')
        latitude=dataset['lat'].values;longitude=dataset['lon'].values
        if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
            raise ValueError('Invalid source georeferencing.')
        indices=[]
        for actual,target in [(latitude,target_latitude),(longitude,target_longitude)]:
            positions=np.abs(actual[:,None]-np.asarray(target)[None]).argmin(0)
            if len(np.unique(positions))!=len(target) or not np.allclose(actual[positions],target,atol=1e-5,rtol=0):
                raise ValueError('Daily source coordinates do not match the fitted MODIS grid.')
            indices.append(positions)
        crop=variable.isel(lat=indices[0],lon=indices[1]).values.astype(np.float32)
        lower=variable.attrs.get('valid_min');upper=variable.attrs.get('valid_max')
        if lower is None or upper is None or not 0<float(lower)<float(upper):
            raise ValueError('Source valid-range metadata is required; do not invent limits.')
        valid=np.isfinite(crop)&(crop>=float(lower))&(crop<=float(upper))&(crop>0)
        clean=np.where(valid,crop,np.nan).astype(np.float32)
        metadata={'product_ids':[path.name],'source':'NASA Aqua MODIS L3m daily chlor_a OCI',
                  'quality_reference':'Source CF decoding/_FillValue and valid_min/valid_max; upstream L3 processing. Individual L2 retrieval flags are unavailable in this mapped file.',
                  'native_qc_applied':True,'units':'mg/m3','valid_min':float(lower),'valid_max':float(upper),
                  'valid_observations':int(valid.sum()),'invalid_observations':int((~valid).sum())}
        metadata.update({name:dataset.attrs.get(name) for name in ['product_name','processing_version','time_coverage_start','time_coverage_end','date_created']})
    return clean,valid,metadata
