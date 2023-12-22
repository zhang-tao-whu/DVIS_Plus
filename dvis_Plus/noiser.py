import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

class Noiser:
    def __init__(self, noise_ratio=0.8, mode='wa'):
        assert mode in ['none', 'rs', 'wa', 'cc', ]
        self.mode = mode
        self.noise_ratio = noise_ratio

    def _rs_noise_forward(self, cur_embeds):
        indices = list(range(cur_embeds.shape[0]))
        np.random.shuffle(indices)
        noise_init = cur_embeds[indices]
        return indices, noise_init

    def _wa_noise_forward(self, cur_embeds):
        # embeds (q, b, c), classes (q)
        indices = list(range(cur_embeds.shape[0]))
        np.random.shuffle(indices)
        noise_init = cur_embeds[indices]
        weight_ratio = torch.rand(cur_embeds.shape[0], 1, 1)
        noise_init = cur_embeds * weight_ratio.to(cur_embeds) + noise_init * (1.0 - weight_ratio.to(cur_embeds))
        ret_indices = torch.arange(cur_embeds.shape[0], dtype=torch.int64).numpy()
        ret_indices[(weight_ratio[:, 0, 0] < 0.5).to(torch.bool).numpy()] =\
            np.array(indices)[(weight_ratio[:, 0, 0] < 0.5).to(torch.bool).numpy()]
        return list(ret_indices), noise_init

    def _cc_noise_forward(self, cur_embeds):
        # embeds (q, b, c), classes (q)
        indices = torch.randint(0, cur_embeds.shape[-1], (cur_embeds.shape[0], )).unsqueeze(-1).unsqueeze(-1)
        weight = torch.arange(cur_embeds.shape[-1], dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        weight = (weight < indices).to(torch.float32).to(cur_embeds)

        indices_, cur_embeds_ = self._rs_noise_forward(cur_embeds)
        ret_embeds = cur_embeds * weight + cur_embeds_ * (1 - weight)
        ret_indices = torch.arange(cur_embeds.shape[0], dtype=torch.int64).numpy()
        ret_indices[(indices[:, 0, 0] < cur_embeds.shape[-1] // 2).to(torch.bool).numpy()] =\
            np.array(indices_)[(indices[:, 0, 0] < cur_embeds.shape[-1] // 2).to(torch.bool).numpy()]
        return list(ret_indices), ret_embeds

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(cur_embds, ref_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    def __call__(self, ref_embeds, cur_embeds, cur_embeds_no_norm=None, activate=False, cur_classes=None):
        if cur_embeds_no_norm is None:
            cur_embeds_no_norm = cur_embeds
        matched_indices = self.match_embds(ref_embeds, cur_embeds)
        if activate and random.random() < self.noise_ratio:
            if self.mode == 'rs':
                indices, noise_init = self._rs_noise_forward(cur_embeds_no_norm)
                return indices, noise_init
            elif self.mode == 'wa':
                indices, noise_init = self._wa_noise_forward(cur_embeds_no_norm)
                return indices, noise_init
            elif self.mode == 'cc':
                indices, noise_init = self._cc_noise_forward(cur_embeds_no_norm)
                return indices, noise_init
            elif self.mode == 'none':
                return matched_indices, cur_embeds_no_norm[matched_indices]
            else:
                raise NotImplementedError
        else:
            return matched_indices, cur_embeds_no_norm[matched_indices]
