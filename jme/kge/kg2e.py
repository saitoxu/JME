import torch
import torch.nn as nn
import torch.nn.functional as F


class KG2E(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str, \
            margin=1.0, vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        self.device = device
        self.margin = margin
        self.ke = dim
        self.vmin = vmin
        self.vmax = vmax

        self.entityEmbedding = nn.Embedding(num_embeddings=entity_size, embedding_dim=dim)
        self.entityCovar = nn.Embedding(num_embeddings=entity_size, embedding_dim=dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_size, embedding_dim=dim)
        self.relationCovar = nn.Embedding(num_embeddings=relation_size, embedding_dim=dim)
        nn.init.xavier_normal_(self.entityEmbedding.weight)
        nn.init.xavier_normal_(self.entityCovar.weight)
        nn.init.xavier_normal_(self.relationEmbedding.weight)
        nn.init.xavier_normal_(self.relationCovar.weight)


    def kl_score(self, relation_m, relation_v, error_m, error_v):
        eps = 1e-9
        losep1 = torch.sum(error_v / (relation_v + eps), dim=1)
        losep2 = torch.sum((relation_m-error_m)**2 / (relation_v + eps), dim=1)
        KLer = (losep1 + losep2 - self.ke) / 2

        losep1 = torch.sum(relation_v / (error_v + eps), dim=1)
        losep2 = torch.sum((error_m - relation_m) ** 2 / (error_v + eps), dim=1)
        KLre = (losep1 + losep2 - self.ke) / 2
        return (KLer + KLre) / 2


    def score(self, inputTriples):
        head, relation, tail = torch.chunk(input=inputTriples, chunks=3, dim=1)
        headm = torch.squeeze(self.entityEmbedding(head), dim=1)
        headv = torch.squeeze(self.entityCovar(head), dim=1)
        tailm = torch.squeeze(self.entityEmbedding(tail), dim=1)
        tailv = torch.squeeze(self.entityCovar(tail), dim=1)
        relationm = torch.squeeze(self.relationEmbedding(relation), dim=1)
        relationv = torch.squeeze(self.relationCovar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        return self.kl_score(relationm, relationv, errorm, errorv)


    def normalize(self):
        ee = self.entityEmbedding.weight
        re = self.relationEmbedding.weight
        ec = self.entityCovar.weight
        rc = self.relationCovar.weight
        ee.weight.data.copy_(torch.renorm(input=ee.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        re.weight.data.copy_(torch.renorm(input=re.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        ec.weight.data.copy_(torch.renorm(input=ec.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        rc.weight.data.copy_(torch.renorm(input=rc.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))


    def forward(self, pos, neg):
        size = pos.size()[0]
        pos_score = self.score(pos)
        neg_score = self.score(neg)
        return torch.sum(F.relu(input=pos_score-neg_score+self.margin)) / size


    def entities(self, e_batch):
        ee = self.entityEmbedding(e_batch)
        ec = self.entityCovar(e_batch)
        return torch.cat([ee, ec], dim=1)


    def relations(self, r_batch):
        re = self.relationEmbedding(r_batch)
        rc = self.relationCovar(r_batch)
        return torch.cat([re, rc], dim=1)
