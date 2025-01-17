#!/usr/bin/env bash

data_dir="data/FB15K-237-50"
model="point"
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
num_epochs=20
num_wait_epochs=50
num_peek_epochs=1
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
beta=0.02
relation_only="False"
relation_only_in_path="False"
relation_path_true="False"
beam_size=256

num_paths_per_entity=-1
margin=-1

hff_dropout_rate=0.1
lff_dropout_rate=0.1
aff_dropout_rate=0.1
relation_dropout_rate=0.2
tailentity_dropout_rate=0.3
rl_module='hrl'
beam_size_high=8
beam_size_low=32
high_embedding_num=4
low_embedding_num=3
#high_attention="True"
#low_attention="True"