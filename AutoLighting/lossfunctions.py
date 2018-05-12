# Parameters being used by loss functions
MU = 50.0
RHO = 50.0
LAMDA = 0.5

def regression_loss_synthetic(syn1, label):
    return (syn1 - label)**2

def feature_loss(syn1, syn2):
    return (syn1 - syn2)**2

def regression_loss(syn1, label):
    rLoss = regression_loss_synthetic(syn1, label)
    loss = RHO * rLoss
    return loss.sum()
