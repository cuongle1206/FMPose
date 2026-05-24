
import torch
import config

def mpjpe(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE), referred to as "Protocol #1".
    """
    assert predicted.shape == target.shape # N x J x C
    return torch.sqrt(((target-predicted)**2).sum(-1)).mean(-1)

def mpjpe_pa(predicted, target):
    """
    Procrusted-Aligned Mean Per-Joint Position Error (MPJPE-PA), referred to as "Protocol #2".
    """
    assert predicted.shape == target.shape # N x J x C
    predicted       = predicted.view((-1,config.N_3DJOINTS,3))
    target          = target.view((-1,config.N_3DJOINTS,3))
    _, _, T, b, c   = procrustes(target, predicted, scaling=True)
    frame_pred      = (b * torch.bmm(predicted, T)) + c
    return torch.sqrt(((target-frame_pred)**2).sum(-1)).mean(-1)

def pck(predicted, target, threshold=150):
    assert predicted.shape == target.shape
    predicted       = predicted.reshape((-1,config.N_3DJOINTS,3))
    target          = target.reshape((-1,config.N_3DJOINTS,3))
    distances       = torch.sqrt(torch.sum((target - predicted)**2, dim=-1))
    pck             = torch.count_nonzero(distances < threshold, dim=-1) / config.N_3DJOINTS
    return pck * 100

def procrustes(X, Y, scaling=True):
    # Reimplementation of MATLAB's `procrustes` function to Pytorch.
    muX     = torch.mean(X, dim=1, keepdim=True) # GT shape N x J x C
    muY     = torch.mean(Y, dim=1, keepdim=True) # Pred shape N x J x C
    X0      = X - muX
    Y0      = Y - muY
    ssX     = torch.sum(torch.sum(X0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)
    ssY     = torch.sum(torch.sum(Y0**2., dim=-1, keepdim=True), dim=-2, keepdim=True)

    # centred Frobenius norm
    normX   = torch.sqrt(ssX)
    normY   = torch.sqrt(ssY)

    # scale to equal (unit) norm
    X0      /= normX
    Y0      /= normY

    # optimum rotation matrix of Y
    A       = torch.bmm(X0.permute(0,2,1), Y0)
    U,s,Vt  = torch.linalg.svd(A,full_matrices=False)
    V       = Vt.permute(0,2,1)
    T       = torch.bmm(V, U.permute(0,2,1))

    V[:,:,-1] *= torch.sign(torch.linalg.det(T)).unsqueeze(1)
    s[:,-1]   *= torch.sign(torch.linalg.det(T))
    T       = torch.bmm(V, U.permute(0,2,1))
    traceTA = torch.sum(s, dim=-1, keepdim=True).unsqueeze(1)

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*torch.bmm(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*torch.bmm(Y0, T) + muX
    c = muX - b*torch.bmm(muY, T)

    return d, Z, T, b, c

def cps(predicted, target):
    assert predicted.shape == target.shape
    predicted       = predicted.reshape((-1,config.N_3DJOINTS,3))
    target          = target.reshape((-1,config.N_3DJOINTS,3))
    _, _, T, b, c   = procrustes(target, predicted, scaling=True)
    frame_pred      = (b * torch.bmm(predicted, T)) + c
    joints_to_use   = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    distances       = torch.sqrt(torch.sum((target[:,joints_to_use] - frame_pred[:,joints_to_use])**2, dim=-1))
    cps             = torch.empty(predicted.shape[0], 300, dtype=torch.double)
    for i, threshold in enumerate(torch.arange(1, 301).tolist()):
        cps[:, i]   = torch.count_nonzero(distances < threshold, dim=-1) == len(joints_to_use)
    return cps

def calculate_symmetry_error(poses, reduction='none', dim=-1, njoints=16):
    bones = {
        'h36m': torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11],
                           [11, 12], [7, 13], [13, 14], [14, 15]], device=poses.device),
        '3dpw': torch.tensor([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
                      [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
                      [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]], device=poses.device)
             }

    # TODO: Fix support for SIMPL joints

    bone_pairs = {'h36m': torch.tensor([[0, 2], [1, 3], [7, 10], [8, 11], [9, 12]], device=poses.device)}
    bone_indices = bones["h36m"] + 1

    start_pos = torch.index_select(poses, -1, bone_indices[:, 0])
    end_pos = torch.index_select(poses, -1, bone_indices[:, 1])

    # Extract bones as delta positions and calculate length
    bone_lengths = torch.linalg.norm(end_pos - start_pos, dim=-2)
    # Find matching bones in skeleton
    bone0 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 0])
    bone1 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 1])

    # Calculate the absolute length difference between symmetries
    absolute_error = torch.abs(bone0 - bone1)

    # Calculate the average error for all bones
    absolute_error = absolute_error.mean(dim=-1)

    if reduction == 'none':
        return absolute_error
    elif reduction == 'mean':
        return absolute_error.mean(dim=dim)
    elif reduction == 'sum':
        return absolute_error.sum(dim=dim)