import torch
import random

class Outputs_Memory_PerClasses:
    def __init__(self, max_len=100,):
        self.class_references = {}
        self.max_len = max_len

    def push(self, references, targets, referecne_match_result):
        # for tracker
        references = references.detach()
        for i in range(len(targets)):
            classes = targets[i]['labels']  # (N, )
            frame_match_result = referecne_match_result[i]
            frame_reference = references[i]
            for i_ref, i_gt in zip(frame_match_result[0], frame_match_result[1]):
                cls = classes[i_gt].item()
                if cls in self.class_references.keys():
                    self.class_references[cls].append(frame_reference[i_ref])
                else:
                    self.class_references[cls] = [frame_reference[i_ref]]
        for cls in self.class_references.keys():
            if len(self.class_references[cls]) > self.max_len:
                self.class_references[cls] = self.class_references[cls][-self.max_len:]
        return

    def push_refiner(self, references, targets, referecne_match_result):
        # for refiner
        references = references.clone().detach()
        classes = targets['labels']  # (N, )
        for i_ref, i_gt in zip(referecne_match_result[0], referecne_match_result[1]):
            cls = classes[i_gt].item()
            if cls in self.class_references.keys():
                self.class_references[cls].extend(list(torch.unbind(references[:, i_ref], dim=0)))
            else:
                self.class_references[cls] = list(torch.unbind(references[:, i_ref], dim=0))

        for cls in self.class_references.keys():
            if len(self.class_references[cls]) > self.max_len:
                random.shuffle(self.class_references[cls])
                self.class_references[cls] = self.class_references[cls][-self.max_len:]
        return

    def get_items(self, cls):
        if cls not in self.class_references.keys():
            return []
        else:
            cls_ref = torch.stack(self.class_references[cls], dim=0)
            return cls_ref

def loss_reid(qd_items, outputs):
    # outputs only using when have not contrastive items
    # compute two loss, contrastive loss & similarity loss
    contras_loss = 0
    aux_loss = 0
    num_qd_items = len(qd_items) # n_instances * frames

    # if none items, return 0 loss
    if len(qd_items) == 0:
        if 'pred_references' in outputs.keys():
            losses = {'loss_reid': outputs['pred_references'].sum() * 0,
                      'loss_aux_reid': outputs['pred_references'].sum() * 0}
        else:
            losses = {'loss_reid': outputs['pred_embds'].sum() * 0,
                      'loss_aux_reid': outputs['pred_embds'].sum() * 0}
        return losses

    for qd_item in qd_items:
        # (n_pos, n_anchor) -> (n_anchor, n_pos)
        pred = qd_item['dot_product'].permute(1, 0)
        label = qd_item['label'].unsqueeze(0)
        # contrastive loss
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])
        # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
        x = torch.nn.functional.pad(
            (_neg_expand - _pos_expand), (0, 1), "constant", 0)
        contras_loss += torch.logsumexp(x, dim=1)

        aux_pred = qd_item['cosine_similarity'].permute(1, 0)
        aux_label = qd_item['label'].unsqueeze(0)
        aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()

    losses = {'loss_reid': contras_loss.sum() / num_qd_items,
              'loss_aux_reid': aux_loss / num_qd_items}
    return losses