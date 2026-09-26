---
dataset_info:
  features:
  - name: pattern_id
    dtype: int32
  - name: example
    dtype: string
  - name: pattern
    dtype: string
  - name: example_type
    dtype:
      class_label:
        names:
          '0': example
          '1': illustration
  - name: synt_type
    dtype: string
  splits:
  - name: train
    num_bytes: 3281947
    num_examples: 15298
  - name: validation
    num_bytes: 606818
    num_examples: 2842
  - name: test
    num_bytes: 600225
    num_examples: 2853
  download_size: 2209154
  dataset_size: 4488990
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
