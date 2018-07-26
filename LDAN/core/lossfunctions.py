# Parameters being used by loss functions
MU = 50.0
RHO = 50.0
LAMDA = 0.5

def regression_loss_synthetic(syn1, label):
    return (syn1 - label)**2

def feature_loss(syn1, syn2):
    return (syn1 - syn2)**2

def regression_loss(syn1, syn2, label):
    rLoss = regression_loss_synthetic(syn1, label) + regression_loss_synthetic(syn2, label)
    fLoss = feature_loss(syn1, syn2)
    loss = RHO * rLoss + LAMDA * fLoss
    return loss.sum()

def regression_loss_2(syn1, label):
    rLoss = regression_loss_synthetic(syn1, label)
    loss = RHO * rLoss
    return loss.sum()

def GAN_loss(vals):
    return vals.mean()
