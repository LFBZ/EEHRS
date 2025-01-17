"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Beam search on the graph.
"""

import torch

import src.utils.ops as ops
from src.utils.ops import unique_max, var_cuda, zeros_var_cuda, int_var_cuda, int_fill_var_cuda, var_to_numpy


def beam_search(pn, e_s, q, e_t, kg, num_steps, beam_size, beam_size_high, beam_size_low, rl_module='hrl', use_relation_tailentity_space_bucketing=True, return_path_components=False):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)
    
    def adjust_search_trace(search_trace, action_offset, r_offset, te_offset):
        for i, x in enumerate(search_trace):
            if type(x) is tuple:
                new_tuple = tuple([_x[r_offset][te_offset][action_offset] for _x in x])
                search_trace[i] = new_tuple

    def top_k_relation(log_r_dist, r_space):
        full_size_high = len(log_r_dist)
        assert (full_size_high % batch_size == 0)
        last_k_high = int(full_size_high / batch_size)  # 1
        r_space_size = r_space.size()[1]  # max_r_space
        beam_r_space_size = log_r_dist.size()[1]  # 10*n
        k_high = min(beam_size_high, beam_r_space_size)  # 4 10*n

        log_r_dist = log_r_dist.view(batch_size, -1)
        log_r_prob, next_r_idx = torch.topk(log_r_dist, k_high)
        next_r = ops.batch_lookup(r_space.view(batch_size, -1), next_r_idx).view(-1)
        # next_r_dist = ops.batch_lookup(r_dist, next_r_idx).view(-1)
        log_r_prob= log_r_prob.view(-1)

        r_beam_offset = next_r_idx // r_space_size  # [batch_size, k_high]
        r_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k_high).unsqueeze(1)  # [batch_size, 1]
        if r_batch_offset.size(0) < r_beam_offset.size(0):
            r_batch_offset = ops.tile_along_beam(r_batch_offset, beam_size=r_beam_offset.size(
                0) // r_batch_offset.size(0))
        r_offset = (r_batch_offset + r_beam_offset).view(-1)
        return next_r, log_r_prob, r_offset

    def top_k_tailentity(log_te_dist, te_space):
        full_size_low = len(log_te_dist)
        assert (full_size_low % batch_size == 0)
        last_k_low = int(full_size_low / batch_size)
        te_space_size = te_space.size()[1]
        beam_te_space_size = log_te_dist.size()[1]
        assert (beam_te_space_size % te_space_size == 0)
        k_low = min(beam_size_low, beam_te_space_size)

        log_te_prob, next_te_idx = torch.topk(log_te_dist, k_low)

        next_te = ops.batch_lookup(te_space, next_te_idx).view(-1)
        # next_te_dist = ops.batch_lookup(te_dist, next_te_idx).view(-1)
        log_te_prob = log_te_prob.view(-1)

        te_beam_offset = next_te_idx // te_space_size
        te_batch_offset = int_var_cuda(torch.arange(full_size_low)).unsqueeze(1)
        if te_batch_offset.size(0) < te_beam_offset.size(0):
            te_batch_offset = ops.tile_along_beam(te_batch_offset, beam_size=te_beam_offset.size(
                0) // te_batch_offset.size(0))
        te_offset = (te_batch_offset + te_beam_offset).view(-1)

        return next_te, log_te_prob, te_offset

    def top_k_action(next_r, next_te, log_r_prob, log_te_prob):
        full_size = len(log_te_prob)
        log_action_prob = (log_r_prob + log_te_prob).view(batch_size, -1)  # [batch_size, beam_size_high*beam_size_low]
        next_r = next_r.view(batch_size, -1)
        next_te = next_te.view(batch_size, -1)  # [batch_size, beam_size_high*beam_size_low]
        # log_r_prob = log_r_prob.view(batch_size, -1)
        # log_te_prob = log_te_prob.view(batch_size, -1)  # [batch_size, beam_size_high*beam_size_low]
        # if r_offset.size(0) < te_offset.size(0):
        #     r_offset = ops.tile_along_beam(r_offset, beam_size=te_offset.size(0) // r_offset.size(0))
        # r_offset = r_offset.view(batch_size, -1)
        # te_offset = te_offset.view(batch_size, -1) # [batch_size, beam_size_high*beam_size_low]
        # (bs, k1 * k2) ==topk==> (bs, k)
        k = min(log_action_prob.size(-1), beam_size)
        log_action_prob, action_idx = torch.topk(log_action_prob, k)
        # next_r = next_r.view(batch_size, -1)
        next_r = next_r[torch.arange(next_r.size(0))[:, None], action_idx].view(-1)
        # next_te = next_te.view(batch_size, -1)
        next_te = next_te[torch.arange(next_te.size(0))[:, None], action_idx].view(-1)
        # log_r_prob = log_r_prob.view(batch_size, -1)
        # log_r_prob = log_r_prob[torch.arange(log_r_prob.size(0))[:, None], action_idx].view(-1)
        # log_te_prob = log_te_prob.view(batch_size, -1)
        # log_te_prob = log_te_prob[torch.arange(log_te_prob.size(0))[:, None], action_idx].view(-1)

        # r_offset = r_offset.view(batch_size, -1)
        # r_offset = r_offset[torch.arange(r_offset.size(0))[:, None], action_idx].view(-1)
        # te_offset = te_offset.view(batch_size, -1)
        # te_offset = te_offset[torch.arange(te_offset.size(0))[:, None], action_idx].view(-1)

        # action_beam_offset = action_idx // next_te.size(1)
        action_batch_offset = int_var_cuda(torch.arange(next_te.size(0))).view(batch_size, -1)
        action_batch_offset = action_batch_offset[torch.arange(action_batch_offset.size(0))[:, None], action_idx].view(-1)
        return (next_r, next_te), log_action_prob, action_batch_offset

    def top_k_answer_unique_action(r_space, e_space, log_r_prob, log_te_prob):
        full_size = len(log_te_prob)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        log_action_dist = (log_r_prob + log_te_prob).view(batch_size, -1)
        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        action_space_size = r_space.size()[1]
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_te_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        action_batch_offset = int_var_cuda(torch.arange(full_size)).view(batch_size, -1)
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            # top_unique_beam_offset = top_unique_idx / action_space_size
            # top_unique_action_offset = top_unique_idx
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_te_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_action_offset = action_batch_offset[i, top_unique_idx].view(-1)
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_te = ops.pad_and_cat(next_te_list, padding_value=kg.dummy_e).view(-1)
        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1).view(-1)
        return (next_r, next_te), log_action_prob, action_offset

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
    d_r = int_fill_var_cuda(e_s.size(), kg.dummy_r)
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)

    inv_q = kg.get_inv_relation_id(q)
    # path_trace = [(inv_q, e_s)]
    init_action = (inv_q, e_s)
    pn.initialize_path((inv_q, e_s), kg)
    # init_action = (r_s, e_s)
    # pn.initialize_path((r_s, e_s), kg)
    d_e = int_fill_var_cuda(e_s.size(), kg.dummy_e)
    pn.update_er_path((d_e, inv_q), kg)
    if kg.args.save_beam_search_paths:
        search_trace = [(inv_q, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1

    # log_action_prob = zeros_var_cuda(batch_size)
    # log_r_prob = zeros_var_cuda(batch_size)
    # log_te_prob = zeros_var_cuda(batch_size * beam_size_high)
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_r_prob = []
        log_te_prob = []
        log_action_probs = []

    action = init_action
    te_offset = None
    action_offset = None

    for t in range(num_steps):
        last_r, e = action

        if e.size(0) != batch_size:
            k_temp = e.size(0) // batch_size
            # => [batch_size * k]
            q_ = ops.tile_along_beam(q, k_temp)
            e_s_ = ops.tile_along_beam(e_s, k_temp)
            e_t_ = ops.tile_along_beam(e_t, k_temp)
        else:
            q_ = q
            e_s_ = e_s
            e_t_ = e_t
        assert (e.size(0) == q_.size(0) == last_r.size(0))

        obs = [e_s_, q_, e_t_, t == (num_steps - 1), last_r, seen_nodes]

        db_outcomes_high, inv_offset_high, relation_entropy = pn.transit_high(
            e, obs, kg, use_relation_tailentity_space_bucketing=use_relation_tailentity_space_bucketing, merge_aspace_batching_outcome=True)
        r_space, r_mask = db_outcomes_high[0][0]
        r_dist = db_outcomes_high[0][1]

        log_r_prob = log_action_prob.view(-1)
        log_r_dist = log_r_prob.view(-1, 1) + ops.safe_log(r_dist)

        if t == num_steps - 1:
            next_r, log_r_prob, r_offset = top_k_relation(log_r_dist, r_space)
        else:
            next_r, log_r_prob, r_offset = top_k_relation(log_r_dist, r_space)

        k_temp = next_r.size(0) // batch_size
        e = e[r_offset].view(-1)
        seen_nodes_ = seen_nodes[r_offset, -1].view(-1)
        last_r_ = next_r
        q_ = ops.tile_along_beam(q, k_temp)
        e_s_ = ops.tile_along_beam(e_s, k_temp)
        e_t_ = ops.tile_along_beam(e_t, k_temp)
        obs = [e_s_, q_, e_t_, t == (num_steps - 1), last_r_, seen_nodes_]

        if t == 0:
            pn.update_er_path((e, next_r), kg, r_offset)
        else:
            pn.update_er_path((e, next_r), kg, te_offset, action_offset, r_offset)

        db_outcomes_low, inv_offset_low, tailentity_entropy = pn.transit_low(
            e, next_r, obs, kg, use_relation_tailentity_space_bucketing=use_relation_tailentity_space_bucketing, merge_aspace_batching_outcome=True)
        te_space, te_mask = db_outcomes_low[0][0]
        te_dist = db_outcomes_low[0][1]

        log_te_dist = ops.safe_log(te_dist)

        if t == num_steps - 1:
            next_te, log_te_prob, te_offset = top_k_tailentity(log_te_dist, te_space)
        else:
            next_te, log_te_prob, te_offset = top_k_tailentity(log_te_dist, te_space)

        next_r = next_r[te_offset].view(-1)
        log_r_prob = log_r_prob[te_offset].view(-1)

        if t == num_steps - 1:
            action, log_action_prob, action_offset = top_k_answer_unique_action(next_r, next_te, log_r_prob, log_te_prob)
            next_te = next_te[action_offset].view(-1)
            d_r = ops.tile_along_beam(d_r, next_te.size(0)//batch_size)
            pn.update_er_path((next_te, d_r), kg, te_offset, action_offset)
            # inv_q = ops.tile_along_beam(inv_q, next_te.size(0)//batch_size)
            # pn.update_er_path((next_te, inv_q), kg, te_offset, action_offset)
        else:
            action, log_action_prob, action_offset = top_k_action(next_r, next_te, log_r_prob, log_te_prob)

        pn.update_path(action, kg, action_offset, r_offset, te_offset)
        seen_nodes = torch.cat([seen_nodes[r_offset][te_offset][action_offset], action[1].unsqueeze(1)], dim=1)

        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)

        if kg.args.save_beam_search_paths:
            adjust_search_trace(search_trace, action_offset, r_offset, te_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
    if kg.args.save_beam_search_paths:
        beam_search_output['search_traces'] = search_trace

    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k+1][0][ind])]
                    e = kg.id2entity[int(search_trace[k+1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = ' <-{}- {} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = ' -{}-> {} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['path_components_list'] = path_components_list

    return beam_search_output


def beam_search_original(pn, e_s, q, e_t, kg, num_steps, beam_size, return_path_components=False):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)

    def top_k_action(log_action_dist, action_space):
        """
        Get top k actions.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*k, action_space_size]
            e_space: [batch_size*k, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)
        # [batch_size, k]
        log_action_prob, action_ind = torch.topk(log_action_dist, k)
        next_r = ops.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
        next_e = ops.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # *** compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind // action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), log_action_prob, action_offset

    def top_k_answer_unique(log_action_dist, action_space):
        """
        Get top k unique entities
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*beam_size, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*beam_size, action_space_size]
            e_space: [batch_size*beam_size, action_space_size]
        :return:
            (next_r, next_e), action_prob, action_offset: [batch_size*k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx // action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
    init_action = (r_s, e_s)
    # path encoder
    pn.initialize_path(init_action, kg)
    if kg.args.save_beam_search_paths:
        search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_action_probs = []

    action = init_action
    for t in range(num_steps):
        last_r, e = action
        assert (q.size() == e_s.size())
        assert (q.size() == e_t.size())
        assert (e.size()[0] % batch_size == 0)
        assert (q.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)
        # => [batch_size*k]
        q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
        e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
        # one step forward in search
        db_outcomes, _, _ = pn.transit(
            e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=True)
        action_space, action_dist = db_outcomes[0]
        # => [batch_size*k, action_space_size]
        log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(action_dist)
        # [batch_size*k, action_space_size] => [batch_size*new_k]
        if t == num_steps - 1:
            action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
        else:
            action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space)
        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)
        pn.update_path(action, kg, action_offset)
        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        if kg.args.save_beam_search_paths:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
    if kg.args.save_beam_search_paths:
        beam_search_output['search_traces'] = search_trace

    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    e = kg.id2entity[int(search_trace[k + 1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = ' <-{}- {} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = ' -{}-> {} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['path_components_list'] = path_components_list

    return beam_search_output
