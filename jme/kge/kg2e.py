import torch
import torch.nn as nn
import torch.nn.functional as F


class KG2E(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str, \
            margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.model = "KG2E"
        self.margin = margin
        self.sim = sim
        self.ke = dim
        self.vmin = vmin
        self.vmax = vmax

        # Embeddings represent the mean vector of entity and relation
        # Covars represent the covariance vector of entity and relation
        self.entityEmbedding = nn.Embedding(num_embeddings=entity_size,
                                            embedding_dim=dim)
        self.entityCovar = nn.Embedding(num_embeddings=entity_size,
                                        embedding_dim=dim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relation_size,
                                              embedding_dim=dim)
        self.relationCovar = nn.Embedding(num_embeddings=relation_size,
                                          embedding_dim=dim)
        nn.init.xavier_normal_(self.entityEmbedding.weight)
        nn.init.xavier_normal_(self.entityCovar.weight)
        nn.init.xavier_normal_(self.relationEmbedding.weight)
        nn.init.xavier_normal_(self.relationCovar.weight)


    def KLScore(self, **kwargs):
        '''
        Calculate the KL loss between T-H distribution and R distribution.
        There are four parts in loss function.
        '''
        eps = 1e-9
        # Calculate KL(e, r)
        losep1 = torch.sum(kwargs["errorv"] / (kwargs["relationv"] + eps), dim=1)
        losep2 = torch.sum((kwargs["relationm"]-kwargs["errorm"])**2 / (kwargs["relationv"] + eps), dim=1)
        KLer = (losep1 + losep2 - self.ke) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(kwargs["relationv"] / (kwargs["errorv"] + eps), dim=1)
        losep2 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / (kwargs["errorv"] + eps), dim=1)
        KLre = (losep1 + losep2 - self.ke) / 2
        return (KLer + KLre) / 2


    def ELScore(self, **kwargs):
        '''
        Calculate the EL loss between T-H distribution and R distribution.
        There are three parts in loss function.
        '''
        eps = 1e-9
        losep1 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / (kwargs["errorv"] + kwargs["relationv"] + eps), dim=1)
        # losep2 = torch.sum(torch.log(kwargs["errorv"]+kwargs["relationv"]), dim=1)
        value = F.relu(kwargs["errorv"]+kwargs["relationv"]) + eps
        losep2 = torch.sum(torch.log(value), dim=1)
        return (losep1 + losep2) / 2


    def scoreOp(self, inputTriples):
        '''
        Calculate the score of triples
        Step1: Split input as head, relation and tail index
        Step2: Transform index tensor to embedding
        Step3: Calculate the score with "KL" or "EL"
        Step4: Return the score
        '''
        head, relation, tail = torch.chunk(input=inputTriples, chunks=3, dim=1)
        headm = torch.squeeze(self.entityEmbedding(head), dim=1)
        headv = torch.squeeze(self.entityCovar(head), dim=1)
        tailm = torch.squeeze(self.entityEmbedding(tail), dim=1)
        tailv = torch.squeeze(self.entityCovar(tail), dim=1)
        relationm = torch.squeeze(self.relationEmbedding(relation), dim=1)
        relationv = torch.squeeze(self.relationCovar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KLScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        elif self.sim == "EL":
            return self.ELScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!" % self.sim)
            exit(1)


    def normalize(self):
        self.entityEmbedding.weight.data.copy_(torch.renorm(input=self.entityEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        self.relationEmbedding.weight.data.copy_(torch.renorm(input=self.relationEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        self.entityCovar.weight.data.copy_(torch.clamp(input=self.entityCovar.weight.detach().cpu(),
                                                       min=self.vmin,
                                                       max=self.vmax))
        self.relationCovar.weight.data.copy_(torch.clamp(input=self.relationCovar.weight.detach().cpu(),
                                                         min=self.vmin,
                                                         max=self.vmax))


    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "entityCovar": self.entityCovar.weight.detach().cpu().numpy(),
                "relationCovar": self.relationCovar.weight.detach().cpu().numpy(),
                "Sim":self.sim}


    def forward(self, posX, negX):
        size = posX.size()[0]

        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        return torch.sum(F.relu(input=posScore-negScore+self.margin)) / size


    def entities(self, e_batch):
        ee = self.entityEmbedding(e_batch)
        ec = self.entityCovar(e_batch)
        return torch.cat([ee, ec], dim=1)


    def relations(self, r_batch):
        re = self.relationEmbedding(r_batch)
        rc = self.relationCovar(r_batch)
        return torch.cat([re, rc], dim=1)
