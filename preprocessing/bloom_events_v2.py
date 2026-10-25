"""Bounded relaxed satellite-event definitions; selection excludes test statistics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
import config
from preprocessing.pipeline import EnvironmentalPipeline,build_sequence_index
from preprocessing.bloom_events import patch_filter,write_json
from preprocessing.data_utils import file_sha256,month_numbers

DEFAULT_DATASET=config.ROOT/'data/processed/bloom_events_v2/dataset_02'
DEFAULT_REPORT=config.ROOT/'docs/results/bloom_events_v2'
CANDIDATES=[
    {'id':'A','percentile':75,'observed_duration':1,'max_unobserved_gap':0,'minimum_patch_pixels':2},
    {'id':'B','percentile':80,'observed_duration':1,'max_unobserved_gap':0,'minimum_patch_pixels':2},
    {'id':'C','percentile':75,'observed_duration':2,'max_unobserved_gap':1,'minimum_patch_pixels':2},
    {'id':'D','percentile':80,'observed_duration':2,'max_unobserved_gap':1,'minimum_patch_pixels':2},
    {'id':'E','percentile':75,'observed_duration':2,'max_unobserved_gap':2,'minimum_patch_pixels':2},
    {'id':'F','percentile':80,'observed_duration':2,'max_unobserved_gap':2,'minimum_patch_pixels':1}]
CRITERION={'minimum_training_components':50,'minimum_validation_components':10,
           'minimum_training_positive_pixel_days':1000,'minimum_validation_positive_pixel_days':100,
           'minimum_validation_positive_fraction':.01}


def relaxed_labels(high,observed,support,definition):
    high,observed,support=np.asarray(high,bool),np.asarray(observed,bool),np.asarray(support,bool)
    if high.shape!=observed.shape or high.ndim!=3 or high.shape[1:]!=support.shape:
        raise ValueError('Incompatible observation/high/support shapes.')
    if np.any(high&~observed) or np.any(observed&~support):raise ValueError('High pixels require actual supported observations.')
    duration,gap,minimum=(definition[k] for k in ('observed_duration','max_unobserved_gap','minimum_patch_pixels'))
    if duration not in (1,2) or gap not in (0,1,2):raise ValueError('Only the bounded 1/2 observed-day, 0–2 gap definitions are supported.')
    qualifying=np.stack([patch_filter(day,minimum) for day in high])
    possible=np.stack([patch_filter(h|(~o&support),minimum) for h,o in zip(high,observed)])
    members=qualifying.copy() if duration==1 else np.zeros_like(qualifying)
    confirmed=members.copy()
    if duration==2:
        for lag in range(1,gap+2):
            pairs=qualifying[:-lag]&qualifying[lag:]
            for offset in range(1,lag):pairs&=~observed[offset:len(observed)-lag+offset]
            members[:-lag]|=pairs;members[lag:]|=pairs
            confirmed[lag:]|=pairs
    targets=np.full(high.shape,255,np.uint8)
    targets[observed&~high]=0
    targets[observed&high&~possible]=0
    if duration==1:targets[observed&high&~qualifying&~possible]=0
    # Unconfirmed high runs and uncertain spatial patches remain unknown.
    targets[members&observed]=1
    asof=np.full_like(targets,255)
    asof[observed&~high]=0
    asof[observed&high&~possible]=0
    asof[confirmed&observed]=1
    return targets,asof


def component_count(positive,observed,max_gap=0):
    """Join daily spatial components across same-pixel observed endpoints only.

    Missing days may link endpoints but are never labelled positive, and an
    observed non-event pixel between endpoints blocks the temporal link.
    """
    layers=np.zeros(positive.shape,np.int32);total=0
    for day in range(len(positive)):
        parts,n=label(positive[day]);parts[parts>0]+=total
        layers[day]=parts;total+=n
    parents=np.arange(total+1)
    def find(item):
        while parents[item]!=item:
            parents[item]=parents[parents[item]];item=int(parents[item])
        return item
    for lag in range(1,max_gap+2):
        valid=positive[:-lag]&positive[lag:]
        for offset in range(1,lag):valid&=~observed[offset:len(observed)-lag+offset]
        if not valid.any():continue
        pairs=np.unique(np.stack([layers[:-lag][valid],layers[lag:][valid]],axis=-1),axis=0)
        for left,right in pairs:
            a,b=find(int(left)),find(int(right))
            if a!=b:parents[b]=a
    return len({find(i) for i in range(1,total+1)})


def summary(targets,observed,gap):
    positive=int((targets==1).sum());valid=int((targets!=255).sum())
    return {'events':component_count(targets==1,observed,gap),'positive_pixel_days':positive,
            'negative_pixel_days':int((targets==0).sum()),'unknown_pixel_days':int((targets==255).sum()),
            'valid_pixel_days':valid,'positive_fraction_of_valid':positive/valid if valid else None,
            'observed_chlorophyll_pixel_days':int(observed.sum()),
            'uncertain_observed_pixel_days':int((observed&(targets==255)).sum())}


def choose_candidate(candidates):
    for row in candidates:
        t,v=row['train'],row['validation']
        if (t['events']>=CRITERION['minimum_training_components'] and v['events']>=CRITERION['minimum_validation_components']
                and t['positive_pixel_days']>=CRITERION['minimum_training_positive_pixel_days']
                and v['positive_pixel_days']>=CRITERION['minimum_validation_positive_pixel_days']
                and v['positive_fraction_of_valid'] is not None
                and v['positive_fraction_of_valid']>=CRITERION['minimum_validation_positive_fraction']):
            return row['definition']
    return None


def prepare(output=DEFAULT_DATASET,report_dir=DEFAULT_REPORT):
    output,report_dir=Path(output),Path(report_dir)
    if output.exists() or report_dir.exists():raise ValueError('Use fresh v2 dataset and report directories.')
    output.mkdir(parents=True);report_dir.mkdir(parents=True)
    protocol={'candidate_order':CANDIDATES,'selection':'First candidate satisfying all training/2024 validation support requirements. No test statistics participate.',
              'criterion':CRITERION,'stop_rule':'If no candidate qualifies, stop without training or further relaxation.',
              'model_selection_protocol':'Same architecture/seed/15-epoch maximum; checkpoint on full 2024 validation loss. Decision/Platt fit uses chronological first-half 2024; second-half validation assesses it. 2025 evaluated once after freeze.',
              'disclosure':'2025 was previously inspected, including the strict run. This is a new, label-statistics-selected proxy definition, not an untouched-test claim.'}
    write_json(report_dir/'protocol.json',protocol)
    p=EnvironmentalPipeline()
    anomaly=np.load(p.label_dir/'chlorophyll_anomaly_2021_2025.npy',mmap_mode='r')
    months=month_numbers(p.dates)
    train=(p.dates>=np.datetime64('2021-01-01'))&(p.dates<=np.datetime64('2023-12-31'))
    thresholds={}
    for q in (75,80):
        values=[]
        for month in range(1,13):
            part=np.asarray(anomaly[train&(months==month)]);valid=part[np.isfinite(part)]
            if not len(valid):raise ValueError('No training anomalies for a calendar month.')
            values.append(float(np.percentile(valid,q)))
        thresholds[q]=np.asarray(values)
    cache={};rows=[]
    for definition in CANDIDATES:
        q=definition['percentile'];row={'definition':definition}
        for split in ('train','validation'):
            first,last=config.SPLITS[split];selected=(p.dates>=np.datetime64(first))&(p.dates<=np.datetime64(last))
            a=np.asarray(anomaly[selected]);observed=np.isfinite(a)&p.ocean
            high=observed&(a>thresholds[q][months[selected]-1,None,None])
            targets,asof=relaxed_labels(high,observed,p.ocean,definition)
            row[split]=summary(targets,observed,definition['max_unobserved_gap'])
            cache[(definition['id'],split)]=(targets,asof)
        rows.append(row)
    chosen=choose_candidate(rows)
    write_json(report_dir/'selection_before_test_labels.json',{'chosen_definition':chosen,'criterion':CRITERION,'train_validation_candidates':rows,'test_statistics_used':False})
    if chosen is None:
        write_json(report_dir/'outcome.json',{'state':'stopped_insufficient_training_validation_support','training_performed':False})
        print('STOP: no bounded relaxed definition meets training/validation support requirements.',flush=True)
        return None
    # Test labels are inspected only after the selection record has been saved.
    first,last=config.SPLITS['test'];selected=(p.dates>=np.datetime64(first))&(p.dates<=np.datetime64(last))
    test_a=np.asarray(anomaly[selected]);test_observed=np.isfinite(test_a)&p.ocean
    chosen_test=None
    for row in rows:
        definition=row['definition'];q=definition['percentile']
        high=test_observed&(test_a>thresholds[q][months[selected]-1,None,None])
        targets,asof=relaxed_labels(high,test_observed,p.ocean,definition)
        count=component_count(targets==1,test_observed,definition['max_unobserved_gap'])
        row['test_label_presence_only']={'positive_labels_exist':bool((targets==1).any()),'event_component_count':count}
        if definition['id']==chosen['id']:chosen_test=(targets,asof)
    write_json(report_dir/'candidate_comparison.json',{'candidates':rows,'chosen_definition':chosen,'test_used_for_selection':False})
    table=['# Bounded relaxed label comparison','',
           '| ID | P | Observed days | Max missing gap | Patch M | Train events | Train positive % | Val events | Val positive % | Test positives? | Test components |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for row in rows:
        d,t,v,test=row['definition'],row['train'],row['validation'],row['test_label_presence_only']
        table.append(f"| {d['id']} | {d['percentile']} | {d['observed_duration']} | {d['max_unobserved_gap']} | {d['minimum_patch_pixels']} | {t['events']} | {100*t['positive_fraction_of_valid']:.4f} | {v['events']} | {100*v['positive_fraction_of_valid']:.4f} | {'yes' if test['positive_labels_exist'] else 'no'} | {test['event_component_count']} |")
    table+=['',f"Chosen: {chosen['id']}, the first qualifying definition. Selection was saved before test label inspection. No model performance was used.",
            '', 'Missing days are never labelled positive and never count toward duration. Components can bridge bounded missing gaps between observed endpoints; they are not independent biological events.']
    (report_dir/'candidate_comparison.md').write_text('\n'.join(table)+'\n');print('\n'.join(table),flush=True)
    if not (chosen_test[0]==1).any():
        write_json(report_dir/'outcome.json',{'state':'stopped_no_positive_test_labels','chosen_definition':chosen,'training_performed':False})
        return None
    labels,asof=np.full(anomaly.shape,255,np.uint8),np.full(anomaly.shape,255,np.uint8);reports={}
    for split,(first,last) in config.SPLITS.items():
        selected=(p.dates>=np.datetime64(first))&(p.dates<=np.datetime64(last))
        targets,causal=chosen_test if split=='test' else cache[(chosen['id'],split)]
        labels[selected]=targets;asof[selected]=causal
        reports[split]={**summary(targets,np.isfinite(anomaly[selected])&p.ocean,chosen['max_unobserved_gap']),'period':[first,last]}
    np.save(output/'event_labels.npy',labels);np.save(output/'confirmed_asof.npy',asof);np.save(output/'support.npy',p.ocean)
    np.save(output/'monthly_thresholds.npy',thresholds[chosen['percentile']])
    index=build_sequence_index(p.dates,labels);index.to_csv(output/'sequence_index.csv',index=False)
    for split in config.SPLITS:reports[split]['windows']=int((index.split==split).sum())
    source_paths=[p.label_dir/name for name in ('label_metadata.json','climatology_metadata.json','monthly_chl_climatology_2021_2025.npy')]+[p.sequence_dir/'preprocessing.json']
    metadata={'schema':'satellite-bloom-events-v2','title':'Satellite bloom-event forecast','chosen_definition':chosen,
              'minimum_patch_pixels':chosen['minimum_patch_pixels'],'duration_days':chosen['observed_duration'],
              'definition':f"Observed daily four-connected chlorophyll-anomaly patches >= {chosen['minimum_patch_pixels']} pixels above training monthly P{chosen['percentile']}; {chosen['observed_duration']} observed high days; bounded missing gap {chosen['max_unobserved_gap']} days.",
              'unknown_policy':'Missing/unobserved cells stay 255. Uncertain sub-patch high cells and unconfirmed multi-day high runs stay unknown. Unknown days never count as observed event days.',
              'fit_period':['2021-01-01','2023-12-31'],'monthly_thresholds':thresholds[chosen['percentile']].tolist(),
              'base_label_dir':str(p.label_dir),'base_sequence_dir':str(p.sequence_dir),
              'fitted_sources':{str(path):file_sha256(path) for path in source_paths},'split_reports':reports,
              'test_disclosure':protocol['disclosure'],'limitations':'Relaxed satellite/chlorophyll-derived proxy, not verified harmful species, toxicity or field HAB occurrence. Single-day patches are not evidence of persistence.',
              'selection_before_test_sha256':file_sha256(report_dir/'selection_before_test_labels.json')}
    write_json(output/'label_metadata.json',metadata)
    stats=deepcopy(p.stats);stats.update(seasonal_features=True,label_metadata_sha256=file_sha256(output/'label_metadata.json'))
    write_json(output/'preprocessing.json',stats)
    files=('event_labels.npy','confirmed_asof.npy','support.npy','monthly_thresholds.npy','sequence_index.csv','label_metadata.json','preprocessing.json')
    write_json(output/'manifest.json',{'schema':'satellite-event-dataset-v2','label_metadata_sha256':file_sha256(output/'label_metadata.json'),
               'sequence_index_sha256':file_sha256(output/'sequence_index.csv'),'environment_fingerprint':p.environment_identity,
               'files':{name:file_sha256(output/name) for name in files}})
    write_json(report_dir/'label_audit.json',metadata)
    write_json(report_dir/'outcome.json',{'state':'labels_ready','chosen_definition':chosen,'training_performed':False})
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT)
    args=parser.parse_args();prepare(args.output_dir,args.report_dir)


if __name__=='__main__':main()
