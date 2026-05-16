# -*- coding:utf-8 -*-
"""
Multi-gate Mixture-of-Experts for mixed regression + classification tasks.

Modelled after the DeepCTR-Torch MMOE implementation.

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al.
        Modeling Task Relationships in Multi-task Learning with
        Multi-gate Mixture-of-Experts[C]
        https://dl.acm.org/doi/10.1145/3219819.3220007
"""
import torch
import torch.nn as nn

from ..basemodel import BaseModel
from ...inputs import combined_dnn_input
from ...layers import DNN, PredictionLayer


class MMOE_MED(BaseModel):
    """Multi-gate Mixture-of-Experts for healthcare multi-task learning.

    Supports one regression task (paid_amount) and one binary classification
    task (RDMIT), or any combination of ``'binary'`` / ``'regression'`` tasks.

    :param dnn_feature_columns: An iterable containing all features used by
        the deep part of the model (SparseFeat and DenseFeat).
    :param num_experts: integer, number of shared expert networks.
    :param expert_dnn_hidden_units: list of positive integers, layer sizes of
        each expert DNN.
    :param gate_dnn_hidden_units: list of positive integers (or empty list),
        layer sizes of each gate DNN.  Empty list means a single linear gate.
    :param tower_dnn_hidden_units: list of positive integers (or empty list),
        layer sizes of the task-specific tower DNN.
    :param l2_reg_linear: float, L2 regularisation strength on the linear part.
    :param l2_reg_embedding: float, L2 regularisation strength on embeddings.
    :param l2_reg_dnn: float, L2 regularisation strength on DNN weights.
    :param init_std: float, std used to initialise embedding vectors.
    :param seed: integer, random seed.
    :param dnn_dropout: float in [0, 1), dropout probability for DNN layers.
    :param dnn_activation: activation function name used in all DNNs.
    :param dnn_use_bn: bool, whether to apply BatchNorm before activation.
    :param task_types: list of str, one per task — ``'binary'`` or
        ``'regression'``.  e.g. ``['regression', 'binary']``.
    :param task_names: list of str, output column names matching the task order.
        e.g. ``['paid_amount', 'RDMIT']``.
    :param device: str, ``'cpu'`` or ``'cuda:0'``.
    :param gpus: list of int or torch.device for multi-GPU.  If None, runs on
        ``device``.  ``gpus[0]`` must match ``device``.

    :return: A PyTorch model instance compatible with the DeepCTR-Torch
        ``.compile()`` / ``.fit()`` / ``.predict()`` API.
    """

    def __init__(
        self,
        dnn_feature_columns,
        num_experts=3,
        expert_dnn_hidden_units=(256, 128),
        gate_dnn_hidden_units=(64,),
        tower_dnn_hidden_units=(64,),
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        l2_reg_dnn=0,
        init_std=1e-4,
        seed=1024,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,
        task_types=("regression", "binary"),
        task_names=("paid_amount", "RDMIT"),
        device="cpu",
        gpus=None,
    ):
        super(MMOE_MED, self).__init__(
            linear_feature_columns=[],
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            device=device,
            gpus=gpus,
        )

        # ── validation ──
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must equal the length of task_types")
        for task_type in task_types:
            if task_type not in ("binary", "regression"):
                raise ValueError(
                    "task must be 'binary' or 'regression', '{}' is illegal".format(task_type)
                )

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        # ── shared expert DNNs ──
        self.expert_dnn = nn.ModuleList([
            DNN(
                self.input_dim, expert_dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                init_std=init_std, device=device,
            )
            for _ in range(self.num_experts)
        ])

        # ── gate DNNs (one per task) ──
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([
                DNN(
                    self.input_dim, gate_dnn_hidden_units,
                    activation=dnn_activation, l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device,
                )
                for _ in range(self.num_tasks)
            ])
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.gate_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )

        self.gate_dnn_final_layer = nn.ModuleList([
            nn.Linear(
                gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0
                else self.input_dim,
                self.num_experts,
                bias=False,
            )
            for _ in range(self.num_tasks)
        ])

        # ── task-specific tower DNNs ──
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList([
                DNN(
                    expert_dnn_hidden_units[-1], tower_dnn_hidden_units,
                    activation=dnn_activation, l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                    init_std=init_std, device=device,
                )
                for _ in range(self.num_tasks)
            ])
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.tower_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )

        self.tower_dnn_final_layer = nn.ModuleList([
            nn.Linear(
                tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0
                else expert_dnn_hidden_units[-1],
                1,
                bias=False,
            )
            for _ in range(self.num_tasks)
        ])

        # ── output activation per task ──
        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        # ── register all DNN weights for regularisation ──
        for module in [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]:
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    module.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )

        self.to(device)

    # ------------------------------------------------------------------
    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # ── expert DNNs ──
        expert_outs = []
        for i in range(self.num_experts):
            expert_outs.append(self.expert_dnn[i](dnn_input))
        expert_outs = torch.stack(expert_outs, 1)   # (bs, num_experts, dim)

        # ── gate DNNs → mixture weights ──
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(
                gate_dnn_out.softmax(1).unsqueeze(1), expert_outs
            )                                        # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze(1))

        # ── task-specific towers ──
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)

        task_outs = torch.cat(task_outs, -1)         # (bs, num_tasks)
        return task_outs