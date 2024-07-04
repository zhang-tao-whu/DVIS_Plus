import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np

from detectron2.projects.point_rend.point_features import point_sample

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class NewInsHungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 num_new_ins: int = 100):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.num_new_ins = num_new_ins

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, prev_frame_indices=None):
        out_prob = outputs['pred_logits'].softmax(-1)  # B, Q, K+1
        B, Q, H, W = outputs['pred_masks'].shape

        indices = []
        for b in range(B):
            valid_inst = targets[b]["valid_inst"]
            new_inst = torch.zeros_like(valid_inst, dtype=torch.bool)
            old_inst = prev_frame_indices[1]
            new_inst[valid_inst] = True
            new_inst[old_inst] = False

            old_src_i, old_tgt_i = prev_frame_indices[0].cpu().numpy(), prev_frame_indices[1].cpu().numpy()
            if torch.all(~new_inst):
                # no new instances appear
                indices.append((old_src_i, old_tgt_i))
                continue

            b_out_prob = out_prob[b]
            tgt_ids = targets[b]["labels"]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -b_out_prob[:, tgt_ids]

            out_mask = outputs['pred_masks'][b]  # Q, H, W
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of  points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )

            C[:, ~new_inst] = 1e+6  # consider newly appeared instances only
            C[:-self.num_new_ins, :] = 1e+6
            src_i, tgt_i = linear_sum_assignment(C.cpu())
            sorted_idx = tgt_i.argsort()
            src_i, tgt_i = src_i[sorted_idx], tgt_i[sorted_idx]

            if len(src_i) == len(new_inst):
                new_tgt_i = tgt_i[new_inst.cpu().numpy()]
                new_src_i = src_i[new_inst.cpu().numpy()]
            else:
                tgt_i_is_new = new_inst.cpu().numpy()[tgt_i]
                new_tgt_i = tgt_i[tgt_i_is_new]
                new_src_i = src_i[tgt_i_is_new]

            assert np.all(new_src_i >= (Q - self.num_new_ins))

            # merge with previous indices
            all_src_i = np.concatenate((old_src_i, new_src_i), axis=0)
            all_tgt_i = np.concatenate((old_tgt_i, new_tgt_i), axis=0)
            indices.append((all_src_i, all_tgt_i))

        return [
            (torch.as_tensor(i, dtype=torch.int64, device=out_prob.device),
             torch.as_tensor(j, dtype=torch.int64, device=out_prob.device))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, prev_frame_indices=None):

        return self.memory_efficient_forward(outputs, targets, prev_frame_indices)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class FrameMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, select_thr):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        indices = []
        aux_indices = []
        valid_masks = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]  num_classes=num_thing+num_stuff
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1)

            valid_inst = targets[b]["valid_inst"]
            C[:, ~valid_inst] = 1e+6

            src_i, tgt_i = linear_sum_assignment(C.cpu())
            sorted_idx = tgt_i.argsort()
            src_i, tgt_i = src_i[sorted_idx], tgt_i[sorted_idx]
            src_i, tgt_i = src_i[valid_inst.cpu().numpy()], tgt_i[valid_inst.cpu().numpy()]
            indices.append((src_i, tgt_i))

            valid_mask = torch.zeros(size=(num_queries,))
            valid_mask[src_i] = 1
            pred_score = torch.max(out_prob[:, :-1], dim=1)[0]
            valid_mask[pred_score > select_thr] = 1
            valid_mask = valid_mask.to(torch.bool)
            valid_masks.append(valid_mask)

            # each output matching with one target
            aux_src_i = torch.arange(num_queries)
            try:
                aux_tgt_i = torch.argmin(C.cpu(), dim=1)
            except:
                aux_tgt_i = torch.full(size=(C.shape[0], ), fill_value=-1).to(aux_src_i)
            aux_tgt_i[src_i] = torch.as_tensor(tgt_i).to(aux_tgt_i)
            aux_tgt_i[~valid_mask] = -1
            aux_indices.append((aux_src_i, aux_tgt_i))

        return [
            (torch.as_tensor(i, dtype=torch.int64, device=device), torch.as_tensor(j, dtype=torch.int64, device=device))
            for i, j in indices
        ], [
            (torch.as_tensor(i, dtype=torch.int64, device=device), torch.as_tensor(j, dtype=torch.int64, device=device))
             for i, j in aux_indices
        ], [valid_mask.to(device) for valid_mask in valid_masks]

    @torch.no_grad()
    def forward(self, outputs, targets, select_thr):
        return self.memory_efficient_forward(outputs, targets, select_thr)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


