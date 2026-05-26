# -*- coding:utf-8 -*-
"""
MoE (Mixture-of-Experts) — single shared gate across all tasks.

Difference from MMoE:
  MMoE: one soft gate *per task*  → each task learns its own expert weighting
  MoE:  one soft gate *shared*    → all tasks use the same expert weighting

This is the intermediate point between SharedBottom (no gating at all) and
MMoE (fully independent gating per task). Useful as a controlled ablation:

  SharedBottom  <--  MoE  <--  MMoE
  (no gating)      (shared)   (per-task)
"""
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer


class MoE(BaseModel):
    """Mixture-of-Experts with a single shared gate (no per-task gating).

    :param dnn_feature_columns: features used by the model.
    :param num_experts: number of expert networks.
    :param expert_dnn_hidden_units: hidden units in each expert DNN.
    :param gate_dnn_hidden_units: hidden units in the (single shared) gate DNN.
    :param tower_dnn_hidden_units: hidden units in each task-specific tower.
    :param task_types: list of 'binary' or 'regression' per task.
    :param task_names: list of task name strings.
    :param device: 'cpu' or 'cuda:0'.
    """

    def __init__(self, dnn_feature_columns, num_experts=8,
                 expert_dnn_hidden_units=(128,),
                 gate_dnn_hidden_units=(64,),
                 tower_dnn_hidden_units=(128, 64),
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'),
                 task_names=('ctr', 'ctcvr'),
                 device='cpu', gpus=None):

        super(MoE, self).__init__(
            linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, seed=seed, device=device, gpus=gpus)

        self.num_tasks   = len(task_names)
        self.num_experts = num_experts
        self.task_names  = task_names
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units   = gate_dnn_hidden_units
        self.tower_dnn_hidden_units  = tower_dnn_hidden_units

        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be > 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be > 1")
        for t in task_types:
            if t not in ('binary', 'regression'):
                raise ValueError(f"task_type must be binary or regression, got {t}")

        self.input_dim = self.compute_input_dim(dnn_feature_columns)

        # Expert DNNs (shared across tasks — same as MMoE)
        self.expert_dnn = nn.ModuleList([
            DNN(self.input_dim, expert_dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                init_std=init_std, device=device)
            for _ in range(num_experts)
        ])

        # ONE shared gate (not one per task — this is the key MoE vs MMoE difference)
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = DNN(self.input_dim, gate_dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn,
                                dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                init_std=init_std, device=device)
            self.gate_final = nn.Linear(gate_dnn_hidden_units[-1], num_experts, bias=False)
        else:
            self.gate_dnn   = None
            self.gate_final = nn.Linear(self.input_dim, num_experts, bias=False)

        # Per-task tower DNNs
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList([
                DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units,
                    activation=dnn_activation, l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device)
                for _ in range(self.num_tasks)
            ])
        else:
            self.tower_dnn = None

        tower_in_dim = (tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0
                        else expert_dnn_hidden_units[-1])
        self.tower_final = nn.ModuleList([
            nn.Linear(tower_in_dim, 1, bias=False)
            for _ in range(self.num_tasks)
        ])

        self.out = nn.ModuleList([PredictionLayer(t) for t in task_types])
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # Expert outputs: (batch, num_experts, expert_dim)
        expert_outs = torch.stack(
            [self.expert_dnn[i](dnn_input) for i in range(self.num_experts)], dim=1)

        # Shared gate: (batch, num_experts)
        if self.gate_dnn is not None:
            gate_out = self.gate_final(self.gate_dnn(dnn_input))
        else:
            gate_out = self.gate_final(dnn_input)
        gate_weights = gate_out.softmax(dim=1)  # (batch, num_experts)

        # Weighted sum of experts: (batch, expert_dim)
        moe_out = torch.matmul(gate_weights.unsqueeze(1), expert_outs).squeeze(1)

        # Task-specific towers — all tasks receive the SAME moe_out
        task_outs = []
        for i in range(self.num_tasks):
            if self.tower_dnn is not None:
                tower_out = self.tower_dnn[i](moe_out)
                logit = self.tower_final[i](tower_out)
            else:
                logit = self.tower_final[i](moe_out)
            task_outs.append(self.out[i](logit))

        return torch.cat(task_outs, dim=-1)