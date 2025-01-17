"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
import os
import pickle

import torch
import torch.nn as nn

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda

from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict


class KnowledgeGraph(nn.Module):
    """
    The discrete knowledge graph is stored with an adjacency list.
    """
    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.adj_list = None
        self.bandwidth = args.bandwidth
        self.args = args
        self.rl_module = args.rl_module

        self.action_space = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.relation_space = None
        self.tailentity_space = None

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        self.max_degree_e = 0
        self.max_degree_er = 0

        print('** Create {} knowledge graph **'.format(args.model))
        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None
        self.set_embedding = args.set_embedding

        if self.set_embedding:
            if self.args.model in ["conve", "complex"]:
                self.define_modules()
                self.initialize_modules()
            elif self.args.model == "point":
                self.define_modules_noRS()
            else:
                self.define_modules_RS()
        else:
            self.define_modules()
            self.initialize_modules()

    def define_modules_RS(self):
        print("Pre-trained embeddings with RS")
        fn_model = self.args.model.split('.')[-1]
        if fn_model == 'conve':
            print("fn model = conve")
            fn_state_dict = torch.load(self.args.conve_state_dict_path)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            print("Embeddings loaded from {}".format(self.args.conve_state_dict_path))
        if fn_model == 'complex':
            print("fn model = complex")
            fn_state_dict = torch.load(self.args.complex_state_dict_path)
            fn_kg_state_dict = get_complex_kg_state_dict(fn_state_dict)
            self.entity_img_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["entity_img_embeddings.weight"], freeze=self.freeze)
            self.relation_img_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["relation_img_embeddings.weight"], freeze=self.freeze)
            print("Embeddings loaded from {}".format(self.args.complex_state_dict_path))
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["entity_embeddings.weight"], freeze=self.freeze)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["relation_embeddings.weight"], freeze=self.freeze)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)
        print(self.entity_embeddings)
        print(self.relation_embeddings)

    def define_modules_noRS(self):
        print("Pre-trained embeddings without RS")
        if self.args.pretrained == "conve":
            fn_model = "conve"
        elif self.args.pretrained == "complex":
            fn_model = "complex"
        if fn_model == 'conve':
            fn_state_dict = torch.load(self.args.conve_state_dict_path)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            print("Embeddings loaded from {}".format(self.args.conve_state_dict_path))
        if fn_model == 'complex':
            print("fn model = {}".format(fn_model))
            fn_state_dict = torch.load(self.args.complex_state_dict_path)
            fn_kg_state_dict = get_complex_kg_state_dict(fn_state_dict)
            self.entity_img_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["entity_img_embeddings.weight"], freeze=self.freeze)
            self.relation_img_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["relation_img_embeddings.weight"], freeze=self.freeze)
            print("Embeddings loaded from {}".format(self.args.complex_state_dict_path))
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["entity_embeddings.weight"], freeze=self.freeze)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding.from_pretrained(fn_kg_state_dict["relation_embeddings.weight"], freeze=self.freeze)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        print('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
        print('Sanity check: {} types loaded'.format(len(self.type2id)))
        with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
        print('Sanity check: {} relations loaded'.format(len(self.relation2id)))
       
        # Load graph structures
        if self.args.model.startswith('point'): 
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
            if self.rl_module == 'original':
                with open(adj_list_path, 'rb') as f:
                    self.adj_list = pickle.load(f)
                    self.vectorize_action_space(data_dir)
            elif self.rl_module == 'eehrs':
                with open(adj_list_path, 'rb') as f:
                    self.adj_list = pickle.load(f)
                    self.vectorize_relation_entity_space(data_dir)

    def vectorize_relation_entity_space(self, data_dir):
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    try:
                        e, score = line.strip().split(':')
                        e_id = self.entity2id[e.strip()]
                        score = float(score)
                        pgrk_scores[e_id] = score
                    except:
                        continue
            return pgrk_scores

        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))

        def get_relation_space(entity):
            relation_space = []
            if entity in self.adj_list:
                relation_space = list(self.adj_list[entity])
                if len(relation_space) + 2 >= self.bandwidth:
                    relation_space = relation_space[:self.bandwidth]
            if entity != DUMMY_ENTITY_ID:
                relation_space.insert(0, NO_OP_RELATION_ID)
            relation_space.insert(0, DUMMY_RELATION_ID)
            return relation_space

        def get_tailentity_space(entity, relation):
            tailentity_space = []
            if entity in self.adj_list:
                if relation in self.adj_list[entity]:
                    tailentity_space = list(self.adj_list[entity][relation])
                    if len(tailentity_space) + 1 >= self.bandwidth:
                        sorted_option_space = sorted(tailentity_space, key=lambda x: page_rank_scores[x],
                                                    reverse=True)
                        tailentity_space = sorted_option_space[:self.bandwidth]
                elif relation == NO_OP_RELATION_ID:
                    tailentity_space.insert(0, entity)
            tailentity_space.insert(0, DUMMY_ENTITY_ID)
            return tailentity_space

        def vectorize_relation_space(relation_space_list, relation_space_size):
            bucket_size = len(relation_space_list)
            r_space = torch.zeros(bucket_size, relation_space_size) + self.dummy_r
            r_mask = torch.zeros(bucket_size, relation_space_size)
            for i, relation_space in enumerate(relation_space_list):
                for j, r in enumerate(relation_space):
                    r_space[i, j] = r
                    r_mask[i, j] = 1
            r_mask[r_space == DUMMY_RELATION_ID] = 0
            return int_var_cuda(r_space), var_cuda(r_mask)

        def vectorize_tailentity_space(tailentity_space_list, tailentity_space_size):
            bucket_size = len(tailentity_space_list)
            te_space = torch.zeros(bucket_size, tailentity_space_size) + self.dummy_e
            te_mask = torch.zeros(bucket_size, tailentity_space_size)
            for i, tailentity_space in enumerate(tailentity_space_list):
                for j, te in enumerate(tailentity_space):
                    te_space[i, j] = te
                    te_mask[i, j] = 1
            te_mask[te_space == DUMMY_ENTITY_ID] = 0
            return int_var_cuda(te_space), var_cuda(te_mask)

        def vectorize_adj_entity_space(adj_entity_space_list, adj_entity_space_size):
            bucket_size = len(adj_entity_space_list)
            adj_space = torch.zeros(bucket_size, adj_entity_space_size) + self.dummy_e
            adj_mask = torch.zeros(bucket_size, adj_entity_space_size)
            for i, adj_entity_space in enumerate(adj_entity_space_list):
                for j, adj_e in enumerate(adj_entity_space):
                    adj_space[i, j] = adj_e
                    adj_mask[i, j] = 1
            adj_mask[adj_space == DUMMY_ENTITY_ID] = 0
            return int_var_cuda(adj_space), var_cuda(adj_mask)

        if self.args.use_relation_tailentity_space_bucketing:
            """
            Store relation tailentity spaces in buckets.
            """
            self.relation_space_buckets = {}
            self.tailentity_space_buckets = {}
            relation_space_buckets_discrete = collections.defaultdict(list)
            tailentity_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            self.entity2relation2bucketid = torch.zeros(self.num_entities, self.num_relations, 2).long()
            self.entity2relation2num = torch.zeros(self.num_entities, self.num_relations).long()

            num_facts_saved_in_relation_table = 0
            num_facts_saved_in_tailentity_table = 0
            for e1 in range(self.num_entities):
                relation_space = get_relation_space(e1)
                key1 = int(len(relation_space) / self.args.relation_bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key1
                self.entity2bucketid[e1, 1] = len(relation_space_buckets_discrete[key1])
                relation_space_buckets_discrete[key1].append(relation_space)
                num_facts_saved_in_relation_table += len(relation_space)

                for r in relation_space:
                    tailentity_space = get_tailentity_space(e1, r)
                    key2 = int(len(tailentity_space) / self.args.tailentity_bucket_interval) + 1
                    self.entity2relation2bucketid[e1, r, 0] = key2
                    self.entity2relation2bucketid[e1, r, 1] = len(tailentity_space_buckets_discrete[key2])
                    self.entity2relation2num[e1, r] = len(tailentity_space)
                    tailentity_space_buckets_discrete[key2].append(tailentity_space)
                    num_facts_saved_in_tailentity_table += len(tailentity_space)

            print('Sanity check: {} facts saved in relation table'.format(
                num_facts_saved_in_relation_table - self.num_entities))
            print('Sanity check: {} facts saved in tailentity table'.format(
                num_facts_saved_in_tailentity_table - self.num_entities))
            for key1 in relation_space_buckets_discrete:
                print('Vectorizing relation spaces bucket {}...'.format(key1))
                self.relation_space_buckets[key1] = vectorize_relation_space(
                    relation_space_buckets_discrete[key1], key1 * self.args.relation_bucket_interval)
            for key2 in tailentity_space_buckets_discrete:
                print('Vectorizing tailentity spaces bucket {}...'.format(key2))
                self.tailentity_space_buckets[key2] = vectorize_tailentity_space(
                    tailentity_space_buckets_discrete[key2], key2 * self.args.tailentity_bucket_interval)

            print("Finish vectorizing relation tailentity space")
        else:

            # sanity check
            num_facts = 0
            out_degrees = collections.defaultdict(int)
            for e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    num_facts += len(self.adj_list[e1][r])
                    out_degrees[e1] += len(self.adj_list[e1][r])
                    if self.max_degree_er < len(self.adj_list[e1][r]):
                        self.max_degree_er = len(self.adj_list[e1][r])

            self.max_degree_e = max([len(self.adj_list[i]) for i in self.adj_list.keys()])
            print("Sanity check: maximum out degree of head entity (maximum relation number of entity): {}".
                         format(self.max_degree_e))
            print("Sanity check: maximum out degree of head entity and relation (maximum relation number "
                         "of entity): {}".format(self.max_degree_er))
            print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
            print('Sanity check: {} facts in knowledge graph'.format(num_facts))

            relation_space_list = []
            tailentity_space_list = []
            max_num_relations = 0
            max_num_tailentity = 0
            for e1 in range(self.num_entities):
                # get relation(option) space
                relation_space = get_relation_space(e1)
                relation_space_list.append(relation_space)
                t = len(relation_space)
                if len(relation_space) > max_num_relations:
                    max_num_relations = len(relation_space)

                # get entity(action) space
                # r2e: dictionary, key: relation idx, value: list of entity idx with same head entity
                r2e = collections.defaultdict()
                if e1 in self.adj_list:
                    for r in self.adj_list[e1]:
                        entity_space = get_tailentity_space(e1, r)
                        r2e[r] = entity_space
                        if len(entity_space) > max_num_tailentity:
                            max_num_tailentity = len(entity_space)
                tailentity_space_list.append(r2e)

            self.relation_space = vectorize_relation_space(relation_space_list, max_num_relations)
            self.tailentity_space = vectorize_tailentity_space(tailentity_space_list, max_num_tailentity)
            print("Finish vectorizing relation entity space")

    def vectorize_action_space(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores
                    
        # Sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        print('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))
        
        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def get_unique_r_space(e1):
            if e1 in self.adj_list:
                return list(self.adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            num_facts_saved_in_action_table = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                action_space_buckets_discrete[key].append(action_space)
                num_facts_saved_in_action_table += len(action_space)
            print('Sanity check: {} facts saved in action table'.format(
                num_facts_saved_in_action_table - self.num_entities))
            for key in action_space_buckets_discrete:
                print('Vectorizing action spaces bucket {}...'.format(key))
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            print('Vectorizing action spaces...')
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)
            
            if self.args.model.startswith('rule'):
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name in ['raw.kb', 'train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                    if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects
       
        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v
        
        self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        self.dev_subject_vectors = answers_to_var(dev_subjects)
        self.dev_object_vectors = answers_to_var(dev_objects)
        self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, 'dev.triples')
        test_path = os.path.join(self.args.data_dir, 'test.triples')
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                print(line)
                if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.adj_list[e1_id]:
                    self.adj_list[e1_id][r_id] = set()
                if not e2_id in self.adj_list[e1_id][r_id]:
                    self.adj_list[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        print('{} fuzzy facts added'.format(count))

        self.vectorize_action_space(self.args.data_dir)

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def virtual_step(self, e_set, r):
        """
        Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
        labeled r and the source entity is in e_set.
        """
        batch_size = len(e_set)
        e_set_1D = e_set.view(-1)
        r_space = self.action_space[0][0][e_set_1D]
        e_space = self.action_space[0][1][e_set_1D]
        e_space = (r_space.view(batch_size, -1) == r.unsqueeze(1)).long() * e_space.view(batch_size, -1)
        e_set_out = []
        for i in range(len(e_space)):
            e_set_out_b = var_cuda(unique(e_space[i].data))
            e_set_out.append(e_set_out_b.unsqueeze(0))
        e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
        return e_set_out

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            if self.args.model == 'complex':
                self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == 'complex':
            self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID        

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
