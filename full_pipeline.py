#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from utils import set_seed
from plot_utils import plot_eval_results

set_seed()


# In[42]:



from mnist_utils import plot_samples
from config import cfg
from vanilla_model import fit as fit_vanilla
from metrics import calc_metrics
from converters import Converter
from config import cfg
from eval import run_tflite_model
from c_utils import write_model_h
from utils import save_test_data
import plot_utils as pu
from model import get_model
from model import save_pruned_model
from utils import get_gzipped_model_size

# In[43]:


from nyuv2_torch_ds_adapter import get_tf_nyuv2_ds
import argparse
args = argparse.Namespace()
args.truncate_testset = False
# args.target_size = (64, 64)
args.crop_size = (640, 480)
# args.target_size = (64, 64)
args.target_size = cfg.img_size
args.out_fold_ratio = 1
args.is_maxim = False
args.batch_size=cfg.batch_size

ds_train, ds_val, ds_test = get_tf_nyuv2_ds(cfg.base_kitti_dataset_dir, args)
# train_size=cfg.take_first_n
# ds_train = ds_train.take(train_size)
# val_size=cfg.take_first_n
# ds_val = ds_val.take(val_size)
# test_size=cfg.take_first_n
# ds_test = ds_test.take(test_size)

x_val= next(iter(ds_val))
x_train= next(iter(ds_train))

# In[44]:


import os
converter = Converter(cfg)
os.makedirs(cfg.save_model_dir, exist_ok=True)
os.makedirs(cfg.save_cfiles_dir, exist_ok=True)
os.makedirs(cfg.save_test_data_dir, exist_ok=True)

# In[46]:


model_names = [
    "depth_model_quant8_dynR",
    "depth_full_quant",
    "depth_qat_int8",
    "pruned_model",
    "pruned_model_unstructured",
    "pruned_model_unstructured_dynamic",
    "pruned_qat_model",
    "depth_model_fp32",
]
cfiles = {
    "depth_model_quant8_dynR": "depth_model_quant8_dynR",
    "depth_full_quant": "q8depth",
    "depth_qat_int8": "qat8depth",
    "pruned_model": "pruned",
    "pruned_model_unstructured": "pruned_unstructured",
    "pruned_model_unstructured_dynamic": "pruned_unstructured_dynamic",
    "pruned_qat_model": "pruned_unstructured_qat_model",
    "depth_model_fp32": "depth_model_fp32",
}
save_test_data(cfg.save_test_data_dir, x_train[0], x_train[1])

# In[50]:


from eval import eval_model
from loss import calculate_loss
from model import get_model
import tensorflow_model_optimization as tfmot


def fit_eval(model, model_name, do_save_model=True):
    keras.backend.clear_session()
    # loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.Mean(name="loss")

    # Define a custom metric
    def custom_metric(y_true, y_pred, sample_weight=None):
        metric_value = calculate_loss(y_true, y_pred)
        metrics.update_state(metric_value, sample_weight=sample_weight)
        return metric_value

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss="mae", metrics=[custom_metric])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.2, min_lr=1e-7, patience=5, min_delta=1e-2, verbose=1
    )
    callbacks = [reduce_lr]
    if "pruned" in model_name:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
    if not cfg.do_overfit:
        tbCallBack = tf.keras.callbacks.TensorBoard(
            log_dir=f"{cfg.logdir}/tb_logs",
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        )
        es = tf.keras.callbacks.EarlyStopping(
            patience=cfg.es_patience, restore_best_weights=True
        )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.logdir + f"/{model_name}",
            save_weights_only=True,
            monitor="custom_metric",
            mode="min",
            save_best_only=True,
        )
        callbacks.append(es)
        callbacks.append(model_checkpoint_callback)
        callbacks.append(tbCallBack)
    history = model.fit(
        x=ds_train,
        epochs=cfg.epochs,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1,
    )
    if do_save_model:
        model.save(f"{cfg.save_model_dir}/{model_name}.h5")
    model_tflite, tflite_path = converter.keras_to_tflite(model, model_name, do_return_path=True)
    # tflite_path='/tmp/models/depth_model_fp32.tflite'
    metrics = eval_model(
        model=model,
        test_ds=ds_val,
        tflite_path=tflite_path,
        model_name=model_name,
        metrics_file_path=f"{cfg.save_model_dir}/metrics.json",
    )
    return model_tflite

# In[51]:


# Build model
fp_model = get_model(
    cfg.img_size, cfg.num_classes, in_channels=cfg.in_channels, use_qat=False
)
fit_eval(fp_model, model_names[7])

# In[19]:


dynR_quant_tflite_model = converter.dynamic_range_quantization(fp_model, model_names[0])

# In[20]:


tflite_model_quant_int8=converter.eight_bit_quantization(fp_model, ds_train, model_name=model_names[1])
converter.check_quantized_model(tflite_model_quant_int8)

# In[23]:


depth_full_quant_tflite_path=f'{cfg.save_model_dir}/{model_names[1]}.tflite'
eval_model(
        ds_val,
        tflite_path=depth_full_quant_tflite_path,
        model=None,
        model_name=model_names[1],
        metrics_file_path=f"{cfg.save_model_dir}/metrics.json",
    )
write_model_h(cfiles[model_names[1]], tflite_model_quant_int8)

# In[29]:


# Convert the model to a quantization aware model
quant_aware_model = get_model(
    cfg.img_size, cfg.num_classes, in_channels=3, use_qat=True
)
model_name = model_names[2]
fit_eval(quant_aware_model, model_name)
quant_aware_model_converted = converter.eight_bit_quantization(
    quant_aware_model, ds_train, model_name=model_name
)
write_model_h(cfiles[model_name], quant_aware_model_converted)

# In[35]:


model_name = model_names[3]
pruned_model = get_model(cfg.img_size, cfg.num_classes, in_channels=3, use_qat=False, use_pruning=True,
                         use_pruning_struct=True)
fit_eval(pruned_model, model_name, do_save_model=False)
pruned_model_for_export = save_pruned_model(pruned_model, f"{cfg.save_model_dir}/{model_name}.h5")
pruned_tflite_model = converter.keras_to_tflite(pruned_model_for_export, model_name)
write_model_h(cfiles[model_name], pruned_tflite_model)

# In[36]:


model_name = model_names[4]
pruned_model_unstruct = get_model(
    cfg.img_size,
    cfg.num_classes,
    in_channels=3,
    use_qat=False,
    use_pruning=True,
    use_pruning_struct=False,
    use_dynamic_sparsity=False,
)
fit_eval(pruned_model_unstruct, model_name, do_save_model=False)
pruned_model_for_export = save_pruned_model(
    pruned_model_unstruct, f"{cfg.save_model_dir}/{model_name}.h5"
)
pruned_tflite_model = converter.keras_to_tflite(pruned_model_for_export, model_name)
write_model_h(cfiles[model_name], pruned_tflite_model)

# In[37]:


model_name = model_names[5]
pruned_model_unstruct_dynamic = get_model(
    cfg.img_size,
    cfg.num_classes,
    in_channels=3,
    use_qat=False,
    use_pruning=True,
    use_pruning_struct=False,
    use_dynamic_sparsity=True,
)
fit_eval(pruned_model_unstruct_dynamic, model_name, do_save_model=False)
pruned_model_unstructured_for_export = save_pruned_model(
    pruned_model_unstruct_dynamic, f"{cfg.save_model_dir}/{model_name}.h5"
)
pruned_tflite_model = converter.keras_to_tflite(pruned_model_unstructured_for_export, model_name)
write_model_h(cfiles[model_name], pruned_tflite_model)

# In[38]:


model_name = model_names[6]
pruned_qat_model = get_model(
    cfg.img_size,
    cfg.num_classes,
    in_channels=3,
    use_qat=True,
    use_pruning=True,
    use_pruning_struct=False,
    use_dynamic_sparsity=False,
    pruned_model_unstructured_for_export=pruned_model_unstructured_for_export
)
fit_eval(pruned_qat_model, model_name, do_save_model=False)
pruned_model_for_export = save_pruned_model(
    pruned_qat_model, f"{cfg.save_model_dir}/{model_name}.h5"
)
pruned_tflite_model = converter.keras_to_tflite(pruned_model_for_export, model_name)
write_model_h(cfiles[model_name], pruned_tflite_model)
