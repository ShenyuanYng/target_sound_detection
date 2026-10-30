# Standalone TSE project

This project owns target sound extraction. SED remains in new_example_code.
Shared source helpers are copied here; never import code from the SED checkout.

## Training storage

Execute `/home/syy12632604/projects/TSE` on the allocated compute node.
All new checkpoints, TensorBoard events, metrics, effective configurations,
scheduler logs and training diagnostics belong under
`/home/syy12632604/projects/TSE/experiments/`, using a distinct run directory.
Do not create code copies in runs or on node-local data disks. Source audio and
reusable audio caches stay on compute nodes. Record Git revision and source
hashes. Local verification may use temporary output directories.

All post-split experiments start from scratch. Resumption is supported only
for runs created by this standalone project. Do not rewrite or migrate old
experiments or their frozen configurations in new_example_code.
