# If tensorflow does not use all GPU memory.
# Note that tensorflow might just use as much memory such that the GPU itself gets saturated and not more.
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4032)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
"""

# Also test intall "TensorRT" from nvidia (go to tensorflow install page).
