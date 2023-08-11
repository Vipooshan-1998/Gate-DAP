import torch
import torch.nn as nn

class Loss(nn.Module):
    def Kl(self, y_true, y_pred):
        eps = 1e-07
        P = y_pred
        P = P / (eps + torch.sum(P, dim=(0, 1, 2, 3), keepdim=True))
        Q = y_true
        Q = Q / (eps + torch.sum(Q, dim=(0, 1, 2, 3), keepdim=True))
        kld = torch.sum(Q * torch.log(eps + Q / (eps + P)), dim=(0, 1, 2, 3))
        # kld=torch.exp(-kld)
        return kld

    def Nss(self, y_true, y_pred):
        """
        Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, 1, h, w) for all tensors.

        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value (one symbolic value per batch element).
        """
        eps = 1e-07
        P = y_pred
        P = P / (eps + torch.max(P))
        Q = y_true

        Qb = torch.round(Q)  # discretize at 0.5
        N = torch.sum(Qb, dim=(0, 1, 2, 3), keepdim=True)
        mu_P = torch.mean(P, dim=(0, 1, 2, 3), keepdim=True)
        std_P = torch.std(P, dim=(0, 1, 2, 3), keepdim=True)
        P_sign = (P - mu_P) / (eps + std_P)

        nss = (P_sign * Qb) / (eps + N)
        nss = torch.sum(nss, dim=(0, 1, 2, 3))

        return -nss  # maximize nss

    def CC(self, y_true, y_pred):
        eps = 1e-07
        P = y_pred
        P = P / (eps + torch.sum(P, dim=(0, 1, 2, 3), keepdim=True))
        Q = y_true
        Q = Q / (eps + torch.sum(Q, dim=(0, 1, 2, 3), keepdim=True))
        N = y_pred.shape[0] * y_pred.shape[2] * y_pred.shape[3] * y_pred.shape[1]
        E_pq = torch.sum(Q * P, dim=(0, 1, 2, 3), keepdim=True)
        E_q = torch.sum(Q, dim=(0, 1, 2, 3), keepdim=True)
        E_p = torch.sum(P, dim=(0, 1, 2, 3), keepdim=True)
        E_q2 = torch.sum(Q ** 2, dim=(0, 1, 2, 3), keepdim=True) + eps
        E_p2 = torch.sum(P ** 2, dim=(0, 1, 2, 3), keepdim=True) + eps
        num = E_pq - ((E_p * E_q) / N)
        den = torch.sqrt((E_q2 - E_q ** 2 / N) * (E_p2 - E_p ** 2 / N))
        return torch.sum(- (num + eps) / (den + eps), dim=(0, 1, 2, 3))  # 相关系数。|cc|<=1, =0 则不相关 1 则正相关， -1 则表示负相关

    def forward(self, y_true, y_pred):
        kl_loss = self.Kl(y_true, y_pred)
        nss_loss = self.Nss(y_true, y_pred)
        cc_loss = self.CC(y_true, y_pred)
        loss = kl_loss + 0.1*nss_loss + 0.1*cc_loss
        # loss = kl_loss
        return loss, kl_loss, nss_loss, cc_loss

# Lossss = Loss()
# a = torch.randn((1,1,224,224))
# # min = torch.min(a)
# # max = torch.max(a)
# # a = (a-min)/(max-min)
# a = a/torch.max(a)
# o = Lossss.Nss(a, a)
# print(o)