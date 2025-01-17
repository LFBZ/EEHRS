#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
if [[ $relation_only_in_path = *"True"* ]]; then
    relation_only_in_path_flag="--relation_only_in_path"
fi
if [[ $relation_path_true = *"True"* ]]; then
    relation_path_true_flag="--relation_path_true"
fi
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi
if [[ $use_relation_tailentity_space_bucketing = *"True"* ]]; then
    use_relation_tailentity_space_bucketing='--use_relation_tailentity_space_bucketing'
fi
if [[ $high_attention = *"True"* ]]; then
    high_attention='--high_attention'
fi
if [[ $low_attention = *"True"* ]]; then
    low_attention='--low_attention'
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    --reward_shaping_threshold $reward_shaping_threshold \
    $relation_only_flag \
    $relation_only_in_path_flag \
    $relation_path_true_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    $use_relation_tailentity_space_bucketing \
    --distmult_state_dict_path $distmult_state_dict_path \
    --complex_state_dict_path $complex_state_dict_path \
    --conve_state_dict_path $conve_state_dict_path \
    --gpu $gpu \
    --rl_module $rl_module \
    --hff_dropout_rate $ff_dropout_rate \
    --lff_dropout_rate $ff_dropout_rate \
    --aff_dropout_rate $ff_dropout_rate \
    --relation_dropout_rate $relation_dropout_rate \
    --tailentity_dropout_rate $tailentity_dropout_rate \
    --beam_size_high $beam_size_high\
    --beam_size_low $beam_size_low\
    --high_embedding_num $high_embedding_num \
    --low_embedding_num $low_embedding_num \
    $high_attention \
    $low_attention \
    $ARGS"

echo "Executing $cmd"

$cmd
