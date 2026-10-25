"""One new full controlled sequence-sampling experiment, development only."""
import numpy as np
import tensorflow as tf
import config
from experiments.train import FeatureGenerator
from models.convlstm import build_convlstm
from training.losses import MaskedWeightedLoss, MaskedAccuracy
from experiments.common import baseline_pipeline, read_json, write_json, collect_predictions, load_model, verify_baseline, utc_now
from experiments.finalize import verify_final, FINAL_DIR
from preprocessing.data_utils import file_sha256

OUT=config.ROOT/'outputs/upgrade/balanced_sampling'


def sampling_distribution(counts):
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=2 or counts.shape[1]!=3 or not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts.sum(0)<=0):
        raise ValueError('Require observed sequence class counts for all three classes.')
    balanced=(counts/counts.sum(0)).mean(1)
    probabilities=.5/len(counts)+.5*balanced
    probabilities=np.minimum(probabilities,3./len(counts))
    return probabilities/probabilities.sum()


class BalancedGenerator(FeatureGenerator):
    def __init__(self,*args,**kwargs):
        self.probabilities=None
        super().__init__(*args,**kwargs)
        if self.split!='train':
            raise ValueError('Balanced sampling is permitted only for training years.')
        self.counts=np.array([[int((self.pipeline.labels[int(r.target_start_idx):int(r.target_end_idx)+1]==c).sum()) for c in range(3)] for r in self.indices.itertuples()])
        self.probabilities=sampling_distribution(self.counts)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.probabilities is None:
            super().on_epoch_end()
        else:
            self.order=self.rng.choice(self.num_samples,self.num_samples,replace=True,p=self.probabilities)


def main():
    verify_baseline(); verify_final()
    if OUT.exists():
        raise ValueError('Sampling experiment already exists; do not repeat or overwrite it.')
    if not (config.ROOT/'outputs/upgrade/strong_diagnostics_2024.json').exists():
        raise ValueError('Complete STRONG diagnostics before training.')
    tf.config.threading.set_intra_op_parallelism_threads(4); tf.config.threading.set_inter_op_parallelism_threads(2)
    pipeline=baseline_pipeline()
    train=BalancedGenerator('train',batch_size=2,shuffle=True,pipeline=pipeline,seasonal=True)
    validation=FeatureGenerator('validation',batch_size=2,shuffle=False,pipeline=pipeline,seasonal=True)
    reference=read_json(FINAL_DIR/'training_config.json')
    record={**reference,'name':'balanced_sampling','created_at_utc':utc_now(),'epochs_completed':0,
            'sampling':'50% uniform + 50% class-balanced sequence mass; cap unnormalized probability at 3/N, renormalize; replacement N draws/epoch.',
            'sampling_seed':42,'only_changed_variable':'training sequence sampling; weights/features/architecture/optimizer/callbacks fixed',
            'sequence_probability_range':[float(train.probabilities.min()),float(train.probabilities.max())],
            'uniform_pixel_forecast_class_counts':train.counts.sum(0).tolist(),
            'expected_sampled_pixel_forecast_class_counts':(train.probabilities@train.counts*train.num_samples).tolist(),
            'retention_criterion':'Under SAME frozen pixel and regional rules: macro F1 gain >=0.005, neither rare-class F1 decreases, regional any/strong recall does not decrease, false warning days do not increase. No post-hoc threshold retuning.'}
    record.pop('model_sha256',None);record.pop('best_epoch',None);record.pop('duration_seconds',None)
    write_json(OUT/'training_config.json',record)
    tf.keras.utils.set_random_seed(42)
    model=build_convlstm(channels=16)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001,clipnorm=1.),loss=MaskedWeightedLoss(reference['class_weights']),metrics=[MaskedAccuracy()])
    import time
    start=time.monotonic()
    history=model.fit(train,validation_data=validation,epochs=30,verbose=2,callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(OUT/'model.keras'),monitor='val_loss',save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5,min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(OUT/'training_history.csv')),tf.keras.callbacks.TerminateOnNaN()])
    record.update(epochs_completed=len(history.history['loss']),best_epoch=int(np.argmin(history.history['val_loss']))+1,
                  duration_seconds=time.monotonic()-start,model_sha256=file_sha256(OUT/'model.keras'))
    write_json(OUT/'training_config.json',record)
    restored=load_model(OUT/'model.keras');p=restored.predict_on_batch(validation[0][0])
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=1e-5):
        raise ValueError('Sampling checkpoint reload failed.')
    collect_predictions(restored,pipeline,OUT/'validation',seasonal=True)
    print('Full sampling training and real checkpoint inference complete.',flush=True)

if __name__=='__main__':
    main()
