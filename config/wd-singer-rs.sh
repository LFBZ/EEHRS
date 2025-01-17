#!/usr/bin/env bash

data_dir="data/WD-singer"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"
use_relation_tailentity_space_bucketing="True"

bandwidth=200
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=100
num_wait_epochs=40
num_peek_epochs=2
batch_size=256
train_batch_size=256
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
beam_size=512

ptranse_state_dict_path="model/WD-singer-PTransE-xavier-200-200-0.001-0.3-0.1/checkpoint-999.tar"
distmult_state_dict_path="model/WD-singer-distmult-xavier-200-200-0.003-0.3-0.1/checkpoint-15.tar"
complex_state_dict_path="model/WD-singer-complex-RV-xavier-200-200-0.003-0.3-0.1/checkpoint-999.tar"
conve_state_dict_path="model/WD-singer-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
tucker_state_dict_path="model/WD-singer-tucker-RV-xavier-200-200-0.0005-32-3-0.3-0.3-0.2-0.1/model_best.tar"

num_paths_per_entity=-1
margin=-1

hff_dropout_rate=0.1
lff_dropout_rate=0.1
aff_dropout_rate=0.1
relation_dropout_rate=0.2
tailentity_dropout_rate=0.3
rl_module='hrl'
beam_size_high=8
beam_size_low=64
high_embedding_num=4
low_embedding_num=3
#high_attention="True"
#low_attention="True"
