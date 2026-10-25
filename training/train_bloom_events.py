"""Fresh satellite bloom-event training, timed pilot and validation-only selection."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
import numpy as np
import tensorflow as tf
import config
from preprocessing.bloom_events import BloomPipeline,DEFAULT_DATASET,write_json
from preprocessing.data_utils import file_sha256
from training.data_generator import HABDataGenerator
from models.bloom_events import build_bloom_event_model,MaskedBinaryLoss

DEFAULT_RUN=config.ROOT/'outputs/bloom_events/run_01'


class BloomGenerator(HABDataGenerator):
    def _load_sample(self,row):
        start,target=int(row['input_start_idx']),int(row['target_start_idx'])
        return self.pipeline.transform(np.arange(start,start+5)),np.asarray(self.pipeline.labels[target:target+5],np.float32)


class RuntimePilot(tf.keras.callbacks.Callback):
    def __init__(self,output,max_epochs):
        super().__init__();self.output=output;self.max_epochs=max_epochs;self.elapsed=[]

    def on_epoch_begin(self,epoch,logs=None):self.start=time.monotonic()

    def on_epoch_end(self,epoch,logs=None):
        self.elapsed.append(time.monotonic()-self.start)
        if epoch==2:
            report={'epochs_timed':3,'epoch_seconds':self.elapsed.copy(),
                    'estimated_seconds_for_max_epochs':float(np.mean(self.elapsed)*self.max_epochs),
                    'maximum_epochs':self.max_epochs,'decision':'Continue the same model/optimizer for at most 15 total epochs if estimated runtime is <=2 hours; otherwise stop after the pilot.',
                    'deadline_budget_seconds':7200,'continuing':bool(np.mean(self.elapsed)*self.max_epochs<=7200)}
            write_json(self.output/'runtime_pilot.json',report)
            print('THREE_EPOCH_RUNTIME '+json.dumps(report),flush=True)
            if not report['continuing']:self.model.stop_training=True


def train(dataset,output,epochs=15):
    output,dataset=Path(output),Path(dataset).resolve()
    if output.exists():raise ValueError('Existing satellite-event runs cannot be overwritten.')
    if epochs<3:raise ValueError('Time at least three epochs before deciding the remaining runtime.')
    tf.keras.utils.set_random_seed(42)
    p=BloomPipeline(dataset)
    train_gen=BloomGenerator('train',2,True,p)
    validation=BloomGenerator('validation',2,False,p)
    first,last=map(np.datetime64,config.SPLITS['train'])
    targets=p.labels[(p.dates>=first)&(p.dates<=last)]
    positive,negative=int((targets==1).sum()),int((targets==0).sum())
    if min(positive,negative)==0:raise ValueError('Training targets need both observed classes.')
    weight=float(np.sqrt(negative/positive))
    model=build_bloom_event_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001,clipnorm=1.),loss=MaskedBinaryLoss(weight))
    output.mkdir(parents=True)
    sources=('preprocessing/bloom_events.py','preprocessing/pipeline.py','preprocessing/data_utils.py',
             'preprocessing/features.py','models/bloom_events.py','models/convlstm.py','models/hab_convlstm.py',
             'training/train_bloom_events.py','training/data_generator.py','config.py','requirements-lock.txt')
    if p.event_metadata.get('schema')=='satellite-bloom-events-v2':
        sources+=('preprocessing/bloom_events_v2.py',)
    record={'schema':'satellite-bloom-event-training-v1','title':'Satellite bloom-event forecast','state':'training',
            'dataset_dir':str(dataset),'dataset_manifest_sha256':file_sha256(dataset/'manifest.json'),
            'seed':42,'input_shape':[5,120,84,16],'output_shape':[5,120,84],'batch_size':2,
            'class_weight_method':'sqrt(training negative/positive pixel-days)',
            'training_class_counts':{'positive':positive,'negative':negative},'positive_weight':weight,
            'learning_rate':.001,'clipnorm':1.,'maximum_epochs':epochs,'checkpoint_selection':'2024 validation masked weighted BCE',
            'train_sequences':train_gen.num_samples,'validation_sequences':validation.num_samples,
            'parameter_count':model.count_params(),'source_sha256':{},
            'dependencies':{name:version(name) for name in ('tensorflow','keras','numpy','scipy','pandas','scikit-learn')},
            'test_disclosure':p.event_metadata['test_disclosure'],'operational_warning_validated':False}
    record['label_definition']=p.event_metadata['definition']
    record['label_schema']=p.event_metadata['schema']
    for name in sources:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(config.ROOT/name,destination);record['source_sha256'][name]=file_sha256(destination)
    write_json(output/'training.json',record)
    pilot=RuntimePilot(output,epochs)
    started=time.monotonic()
    history=model.fit(train_gen,validation_data=validation,epochs=epochs,verbose=2,callbacks=[
        pilot,tf.keras.callbacks.ModelCheckpoint(str(output/'model.keras'),monitor='val_loss',save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(output/'history.csv')),tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,start_from_epoch=2,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5,min_lr=1e-5)])
    if not all(np.isfinite(history.history[key]).all() for key in ('loss','val_loss')):
        raise ValueError('Nonfinite loss; do not serve this run.')
    restored=tf.keras.models.load_model(output/'model.keras',compile=False)
    X,y=validation[0];prediction=restored(X,training=False).numpy()
    if prediction.shape!=y.shape or not np.isfinite(prediction).all():raise ValueError('Checkpoint reload failed.')
    record.update(state='trained',epochs_completed=len(history.history['loss']),best_epoch=int(np.argmin(history.history['val_loss']))+1,
                  duration_seconds=time.monotonic()-started,model_sha256=file_sha256(output/'model.keras'))
    write_json(output/'training.json',record);return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--epochs',type=int,default=15)
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    print(json.dumps(train(args.dataset_dir,args.output_dir,args.epochs),indent=2))


if __name__=='__main__':main()
