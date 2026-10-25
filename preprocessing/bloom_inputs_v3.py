"""Causal input-only ablation wrapping the preserved run_02 dataset."""
import json
from pathlib import Path
import numpy as np
from preprocessing.bloom_events import BloomPipeline
from preprocessing.data_utils import file_sha256
import config

BASE=config.ROOT/'data/processed/bloom_events_v2/dataset_02'
ROOT=config.ROOT/'data/processed/bloom_events_v3'
GROUPS={
 'V0':[],
 'V1':['last_event','last_event_known','observation_age'],
 'V2':['last_event','last_event_known','observation_age','chl_change','chl_change_valid','sst_front','sst_front_valid',
       'wind_speed','wind_direction_u','wind_direction_v','wind_valid','wind_history_u','wind_history_v','wind_history_valid'],
}
GROUPS['V3']=GROUPS['V2']+['distance_to_coast_km']
FLAGS={'last_event','last_event_known','chl_change_valid','sst_front_valid','wind_direction_u','wind_direction_v','wind_valid','wind_history_valid'}

def coastal_distance(latitude,longitude,geojson):
    """Physical polygon coast distance, local equirectangular km approximation; land=0."""
    from matplotlib.path import Path as PolygonPath
    xx,yy=np.meshgrid(longitude,latitude)
    points=np.stack([xx.ravel(),yy.ravel()],axis=-1)
    reference=float(np.mean(latitude));factor=np.array([111.195*np.cos(np.deg2rad(reference)),111.195])
    projected=points*factor
    squared=np.full(len(points),np.inf);on_land=np.zeros(len(points),bool)
    for feature in geojson['features']:
        geometry=feature['geometry']
        polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
        for polygon in polygons:
            outer=np.asarray(polygon[0],float)
            if (outer[:,0].max()<min(longitude)-3 or outer[:,0].min()>max(longitude)+3
                    or outer[:,1].max()<min(latitude)-3 or outer[:,1].min()>max(latitude)+3):continue
            inside=PolygonPath(outer).contains_points(points)
            for hole in polygon[1:]:inside&=~PolygonPath(np.asarray(hole)).contains_points(points)
            on_land|=inside
            for ring in polygon:
                vertices=np.asarray(ring,float)*factor
                for left,right in zip(vertices[:-1],vertices[1:]):
                    vector=right-left;length=float(vector@vector)
                    fraction=np.clip(((projected-left)@vector)/max(length,1e-12),0,1)
                    distance=np.sum((projected-left-fraction[:,None]*vector)**2,axis=1)
                    squared=np.minimum(squared,distance)
    if not np.isfinite(squared).all():raise ValueError('No usable physical coast segments in local reference.')
    result=np.sqrt(squared);result[on_land]=0
    return result.reshape(len(latitude),len(longitude)).astype(np.float32)


def daily_channels(p,day,previous,last_seen,coast):
    """Only indices <= day are accessed. History resets at chronological split start."""
    observed=p.asof[day]!=255
    state=np.where(observed,p.asof[day],previous)
    seen=np.where(observed,day,last_seen)
    known=seen>=0
    age=np.where(known,day-seen,np.nan).astype(float)
    year=str(p.dates[day])[:4]
    first=max(0,day-4)
    while first<day and str(p.dates[first])[:4]!=year:first+=1
    indices=np.arange(first,day+1)
    chl,cv=p.valid('chlorophyll',indices)
    count=cv.sum(0);total=np.where(cv,chl,0).sum(0)
    latest=np.zeros(p.ocean.shape,float);last=np.full(p.ocean.shape,-1)
    for i in range(len(indices)):
        latest[cv[i]]=chl[i][cv[i]];last[cv[i]]=i
    change_valid=count>=2
    change=np.where(change_valid,latest-total/np.maximum(count,1),np.nan)
    sst,sv=p.valid('sst',np.array([day]));sst=np.where(sv[0],sst[0],np.nan)
    gy,gx=np.gradient(sst,np.asarray(p.latitude)*111.,np.asarray(p.longitude)*111.*np.cos(np.deg2rad(np.mean(p.latitude))))
    front=np.hypot(gy,gx);front_valid=np.isfinite(front)&p.ocean
    u,uv=p.valid('wind_u',indices);v,vv=p.valid('wind_v',indices);wv=uv&vv
    speed=np.hypot(u[-1],v[-1]);wind_valid=wv[-1]
    direction_valid=wind_valid&(speed>1e-8)
    history_count=wv.sum(0);history_valid=history_count>0
    hu=np.where(wv,u,0).sum(0)/np.maximum(history_count,1)
    hv=np.where(wv,v,0).sum(0)/np.maximum(history_count,1)
    channels={'last_event':np.where(known,state,0),'last_event_known':known,'observation_age':age,
      'chl_change':change,'chl_change_valid':change_valid,
      'sst_front':np.where(front_valid,front,np.nan),'sst_front_valid':front_valid,
      'wind_speed':np.where(wind_valid,speed,np.nan),
      'wind_direction_u':np.where(direction_valid,u[-1]/np.maximum(speed,1e-8),0),
      'wind_direction_v':np.where(direction_valid,v[-1]/np.maximum(speed,1e-8),0),
      'wind_valid':direction_valid,'wind_history_u':np.where(history_valid,hu,np.nan),
      'wind_history_v':np.where(history_valid,hv,np.nan),'wind_history_valid':history_valid,
      'distance_to_coast_km':coast}
    return channels,state,seen


def prepare():
    ROOT.mkdir(exist_ok=False)
    p=BloomPipeline(BASE)
    coast_path=config.ROOT/'outputs/roadmap_20261015/reference/land.geojson'
    coast=coastal_distance(p.latitude,p.longitude,json.loads(coast_path.read_text()))
    np.save(ROOT/'coast_km.npy',coast)
    names=GROUPS['V3']
    raw=np.lib.format.open_memmap(ROOT/'features.npy',mode='w+',dtype=np.float32,shape=(*p.shape,len(names)))
    state=np.zeros(p.ocean.shape,np.uint8);seen=np.full(p.ocean.shape,-1)
    for day in range(len(p.dates)):
        if day==0 or str(p.dates[day])[:4]!=str(p.dates[day-1])[:4]:
            state[:]=0;seen[:]=-1
        channels,state,seen=daily_channels(p,day,state,seen,coast)
        for c,name in enumerate(names):raw[day,:,:,c]=channels[name]
    train=p.dates<=np.datetime64('2023-12-31')
    train&=p.dates>=np.datetime64('2021-01-01')
    stats={}
    for c,name in enumerate(names):
        values=np.asarray(raw[:np.flatnonzero(train)[-1]+1,:,:,c]);valid=np.isfinite(values)&p.ocean
        observed=values[valid]
        if name in FLAGS:record={'kind':'flag_or_unit_direction','median':0.,'scale':1.,'minimum':-1.,'maximum':1.}
        else:
            q01,q25,q50,q75,q99=np.percentile(observed,[1,25,50,75,99])
            record={'kind':'train_robust','median':float(q50),'scale':float(max(q75-q25,1e-6)),'minimum':float(q01),'maximum':float(q99)}
            if name=='observation_age':record.update(minimum=0.,maximum=float(max(1,np.percentile(observed,95))),median=0.,scale=float(max(1,np.percentile(observed,95))))
        stats[name]=record
        for day in range(len(p.dates)):
            a=raw[day,:,:,c];scaled=(np.clip(a,record['minimum'],record['maximum'])-record['median'])/record['scale']
            scaled[~np.isfinite(a)|~p.ocean]=0;raw[day,:,:,c]=scaled
    raw.flush()
    record={'fit_period':['2021-01-01','2023-12-31'],'channels':names,'groups':GROUPS,'statistics':stats,
      'base_dataset':str(BASE),'base_manifest_sha256':file_sha256(BASE/'manifest.json'),
      'features_sha256':file_sha256(ROOT/'features.npy'),'coast_source':str(coast_path),'coast_source_sha256':file_sha256(coast_path),
      'bathymetry':'Skipped: no existing named local depth/bathymetry grid found; no downloads.',
      'history':'State and age carry past observations within each calendar year; reset yearly. Each timestep dynamics use its trailing <=5 observed-calendar-day window, no future. Origin summary uses exact five input days.',
      'v4':'Skipped: optional stretch excluded to preserve time for four controlled comparisons.',
      'targets':'Unmodified run_02 P75 one-day patch>=2 no-gap labels. Unknown=255.'}
    (ROOT/'features.json').write_text(json.dumps(record,indent=2)+'\n')
    # Target and index files remain in the protected dataset; wrapper verifies its identity.
    return record


class InputPipeline(BloomPipeline):
    def __init__(self,dataset=BASE,variant='V0'):
        super().__init__(dataset)
        if variant not in GROUPS:raise ValueError('Unknown predeclared input variant.')
        self.variant=variant
        self.feature_record=json.loads((ROOT/'features.json').read_text())
        if file_sha256(BASE/'manifest.json')!=self.feature_record['base_manifest_sha256']:raise ValueError('Run_02 dataset changed.')
        self.extra=np.load(ROOT/'features.npy',mmap_mode='r')
        self.columns=[GROUPS['V3'].index(name) for name in GROUPS[variant]]
    def transform(self,indices):
        base=super().transform(indices)
        if not self.columns:return base
        return np.concatenate([base,np.asarray(self.extra[indices])[...,self.columns]],axis=-1).astype(np.float32)
