
model:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3

training:
  lr: 0.001
  weight_decay: 1e-5
  epochs: 100
  batch_size: 64
  lambda_cls: 1.0
  lambda_reg: 0.5
  lambda_temp: 0.3
  lambda_edge: 0.3
  lambda_hier: 0.3

data:
  path: ./data/CTU-13
  temporal_window: 300
  augment_ratio: 0.2
  restart_prob: 0.3
