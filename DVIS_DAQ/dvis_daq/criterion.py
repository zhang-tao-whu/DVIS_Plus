import random

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class DAQCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, num_new_ins):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.num_new_ins = num_new_ins

    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = []
        target_classes = []
        for output_i, target_i, indices_i in zip(outputs, targets, indices):
            pred_logits = output_i["pred_logits"][0]  # q, k+1
            tgt_classes_o = target_i[0]["labels"]
            valid_inst = target_i[0]["valid_inst"][indices_i[0][1]]
            for disappear_gt_id in output_i["disappear_tgt_ids"]:
                # assert ~target_i[0]["valid_inst"][disappear_gt_id]
                valid_inst[indices_i[0][1] == disappear_gt_id] = False
            if pred_logits.shape[0] == 0:
                continue

            src_idx, tgt_idx = indices_i[0][0][valid_inst], indices_i[0][1][valid_inst]
            assert len(src_idx) == torch.sum(valid_inst)

            tgt_classes = torch.full(
                pred_logits.shape[:1], self.num_classes, dtype=torch.int64, device=pred_logits.device
            )
            tgt_classes[src_idx] = tgt_classes_o[tgt_idx]

            src_logits.append(pred_logits)
            target_classes.append(tgt_classes)

        if len(src_logits) == 0:
            loss_ce = outputs[0]["pred_logits"].sum() * 0.0
            losses = {"loss_ce": loss_ce}
            return losses

        src_logits = torch.cat(src_logits, dim=0).unsqueeze(0)  # b, q, k+1
        target_classes = torch.cat(target_classes, dim=0).unsqueeze(0)  # b, q

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        if torch.isnan(loss_ce):
            loss_ce = outputs[0]["pred_logits"].sum() * 0.0
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_masks = []
        target_masks = []
        for output_i, target_i, indice_i in zip(outputs, targets, indices):
            valid_inst = target_i[0]["valid_inst"][indice_i[0][1]]
            for disappear_gt_id in output_i["disappear_tgt_ids"]:
                # assert ~target_i[0]["valid_inst"][disappear_gt_id]
                valid_inst[indice_i[0][1] == disappear_gt_id] = False
            # valid_inst[indice_i[0][1] == output_i["disappear_tgt_id"]] = False
            src_idx, tgt_idx = indice_i[0][0][valid_inst], indice_i[0][1][valid_inst]

            pred_masks = output_i["pred_masks"][0][src_idx]  # ntgt, h, w
            tgt_masks = target_i[0]["masks"][tgt_idx]  # ntgt, h, w

            src_masks.append(pred_masks)
            target_masks.append(tgt_masks)

        src_masks = torch.cat(src_masks, dim=0)
        target_masks = torch.cat(target_masks, dim=0).to(src_masks)
        N = src_masks.shape[0]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        point_logits = point_logits.view(N, self.num_points)
        point_labels = point_labels.view(N, self.num_points)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        assert len(outputs) == len(targets)

        indices = [out["indices"] for out in outputs]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = 0
        for _indices in indices:
            num_masks += len(_indices[0][1])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs[0].values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks)
            )

        if "aux_outputs" in outputs[0]:
            len_aux = len(outputs[0]["aux_outputs"])
            for i in range(len_aux):
                aux_outputs = [outputs_i["aux_outputs"][i] for outputs_i in outputs]

                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
