import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class AllLoss(nn.Module):
    def __init__(self, view, alphas, batch_size) -> None:
        super(AllLoss, self).__init__()
        self.view = view
        self.alphas = alphas
        self.orthLoss = OrthogonalLoss()
        self.contrastiveLoss = ViewContrastiveLoss(batch_size, 0.5, 1.0)
        self.userInterLoss = UserInnerContrastiveLoss(0.5)

    # 使用 user inner loss
    def forwardSharedAndSpecific(self, view_specific, view_shared):

        user_inner_loss = 0

        contrastiveLoss = 0
        for i in range(self.view):
            for j in range(i+1, self.view):
                # contrastiveLoss += self.contrastiveLoss(view_coms[i], view_coms[j])
                contrastiveLoss += self.contrastiveLoss(view_specific[i], view_specific[j])
                user_inner_loss += self.userInterLoss(view_shared[i], view_specific[i], view_shared[j], view_specific[j])
        
        loss = contrastiveLoss * self.alphas["contrastive"] + user_inner_loss * self.alphas["user_inner"]
        return loss

    def forward(self, view_specific, view_shared):
        loss_shared = self.forwardSharedAndSpecific(view_specific, view_shared)

        orthogonalLoss = self.orthLoss(view_shared, view_specific)

        loss = loss_shared + orthogonalLoss * self.alphas["orth"]
        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    # Should be orthogonal
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    def forward(self, shared_output,specific_output):
        num_view = len(specific_output)

        orthogonal_loss = None
        # print("shared ",len(shared_output))
        # print("specific", len(specific_output))
        # print(num_view)
        for i in range(num_view):
            if torch.is_tensor(shared_output):
                shared = shared_output
            else:
                # print("shared veiw arr")
                shared = shared_output[i]
            specific = specific_output[i]
            loss = self.orthogonal_loss(shared, specific)
            if orthogonal_loss is None:
                orthogonal_loss = loss
            else:
                orthogonal_loss += loss 

        # print("orthogonal loss is {}, similarity_loss is {}".format(orthogonal_loss, similarity_loss))
        return orthogonal_loss
    
class AdverseLoss(nn.Module):
    def __init__(self, device) -> None:
        super(AdverseLoss, self).__init__()
        self.device = device
    
    def forward(self, z_hats):
        loss = 0
        for i, z_hat in enumerate(z_hats):
            z = torch.full((z_hat.size(dim=0), 1), i)
            z = z.to(self.device)
            z = z.squeeze()
            # print("z", z.size())
            # print("z_hat", z_hat.size())
            z_l = F.cross_entropy(z_hat, z)
            loss += torch.exp(-z_l)
        return loss

class ViewContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature_f, temperature_l):
        super(ViewContrastiveLoss, self).__init__()
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l


        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, h_i, h_j):
        batch_size = h_i.size()[0]
        N = 2 * batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class UserInnerContrastiveLoss(nn.Module):
    def __init__(self, temperature) -> None:
        super(UserInnerContrastiveLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, shared_i, specific_i, shared_j, specific_j):
        batch_size = shared_i.size()[0]

        # shared_i <-> shared_j
        matrix_shared_shared = torch.matmul(shared_i, shared_j.T) / self.temperature
        sim_shared_shared = torch.diag(matrix_shared_shared)
        positive_samples = sim_shared_shared.reshape(batch_size,1)

        # shared_i <-> specific_j
        matrix_shared_specific = torch.matmul(shared_i, specific_j.T) / self.temperature
        sim_shared_specific = torch.diag(matrix_shared_specific)
        negative_samples_1 = sim_shared_specific.reshape(batch_size, 1)

        # specific_i <-> shared_j
        matrix_spcific_shared = torch.matmul(specific_i, shared_j.T) / self.temperature
        sim_specific_shared = torch.diag(matrix_spcific_shared)
        negative_samples_2 = sim_specific_shared.reshape(batch_size, 1)

        # specific <-> specific
        matrix_specific_specific = torch.matmul(specific_i, specific_j.T) / self.temperature
        sim_specific_specific = torch.diag(matrix_specific_specific)
        negative_samples_3 = sim_specific_specific.reshape(batch_size, 1)

        negative_samples = torch.cat([negative_samples_1, negative_samples_2, negative_samples_3], dim=1)
        
        lables  = torch.zeros(batch_size).to(shared_i.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        loss = self.criterion(logits, lables) / batch_size
        return loss