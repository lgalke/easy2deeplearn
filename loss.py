class NTXent(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.sim = nn.CosineSimilarity(dim=-1)
        self.xent = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, z_candidates, z_msg, targets=None):
        scores = self.sim(z_candidates, z_msg)
        # TODO merge in temperature
        if targets is None:
            return scores

        loss = self.xent(scores.unsqueeze(0), targets)

        return scores, loss
