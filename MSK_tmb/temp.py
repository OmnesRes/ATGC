# (index, (sample level inputs), (sample label, and stratification))
ds = tf.data.Dataset.from_tensor_slices((idx_sample, (x_sample1, x_sample2, etc), (y_label, y_strat)))
# then you should be able to batch (either with standard or custom)
ds = ds.batch(10)
# then load
instance_loader1
instance_loader2
# creating a final tuple of inputs, y, and optional weights (not included here)

ds = tf.data.Dataset.from_tensor_slices((idx_sample, (x_sample1, x_sample2, etc), (y_label, y_strat)))
ds = ds.map(lambda idx, x, y: (((instance_loader1(idx), instance_loader2(idx)) + x), y))



ds = tf.data.Dataset.from_tensor_slices((idx_sample, x_sample1, x_sample2, etc, y_label, y_strat))
ds = ds.map(lambda *args: (((instance_loader1(args[0], to_ragged=[True]), etc., ) + args[1: -1]), args[-1]))






