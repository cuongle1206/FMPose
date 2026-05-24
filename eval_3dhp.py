# Cuong Le - CVL, Linköping University
import os, shutil
import pickle, math
import numpy as np
import torch

# database
import data.data_3dhp as dataset
import config as c
from utils.data import *
from utils.metrics import *
from models.model import FlowNet, WrappedModel
from alive_progress import alive_bar

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from functools import partial

# logging
from prettytable import PrettyTable
import wandb

def test(mode='all'):
    print()
    print("Program is running on: ", c.device)
    print("Evaluating experiment: ", c.args.exp, "\n")
    
    if mode == 'all': dataset = testset
    elif mode == 'GS': dataset = GSset
    elif mode == 'noGS': dataset = noGSset
    elif mode == 'out': dataset = outdoorset
    else: raise ValueError("Invalid mode")
    print("-- on 3DHP testset -- mode: ", mode)
    
    test_loader         = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, shuffle=False, drop_last=False)
    odefunc.load(c.model_best, c.device)
    odefunc.eval()
    
    total_iters         = math.ceil(len(dataset)/c.batch_size)
    total_err_zero_p1, total_err_zero_p2, total_pck_zero        = [], [], []
    total_err_mean_p1, total_err_mean_p2, total_pck_mean        = [], [], []
    total_err_worst_p1, total_err_worst_p2, total_pck_worst     = [], [], []
    total_err_best_p1, total_err_best_p2, total_pck_best        = [], [], []
    total_err_median_p1, total_err_median_p2, total_pck_median  = [], [], []
    
    with alive_bar(total=total_iters, title='Testing', length=10, bar='bubbles', spinner='elements', refresh_secs=0.3) as bar:
        for _, batch in enumerate(test_loader):
            p2d_k           = batch["p2d_topk"].to(c.device)                # N x 16 x 2 x k
            p2d_max         = batch["p2d"].to(c.device)                     # N x 16 x 2
            p3d             = batch["p3d"].to(c.device)
            p3d_gt          = batch["p3d_gt"]
            N, J, C         = p3d.shape
            
            T               = torch.linspace(0,1,c.n_steps).to(c.device)  # sample times
            x0              = torch.zeros_like(p3d)
            if c.args.sampling:
                k_indices       = torch.randperm(96)[:c.args.topk]
                p2d             = torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1) # N x 16 x 2 x (1+k)
                odefunc_ags     = partial(odefunc, p2d=p2d[..., torch.randperm(c.args.topk+1)])
            else:
                odefunc_ags     = partial(odefunc, p2d=p2d_k[..., torch.randperm(c.args.topk)])
            solver          = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
            x1              = solver.sample(time_grid=T, x_init=x0, method='midpoint', step_size=c.step_size, return_intermediates=False)
            x1              = x1.detach().cpu()
            x1_ra           = (x1 - x1[:,0:1,:])*1e3 # [m] -> [mm]
            total_err_zero_p1.append(mpjpe(x1_ra, p3d_gt))
            total_err_zero_p2.append(mpjpe_pa(x1_ra, p3d_gt))
            total_pck_zero.append(pck(x1_ra, p3d_gt))
            
            x0_all          = c.std_dev * torch.randn_like(p3d.unsqueeze(0).repeat(c.n_hypo,1,1,1))
            if c.args.sampling:
                p2d_hypos       = []
                for _ in range(c.n_hypo):
                    k_indices   = torch.randperm(96)[:c.args.topk]
                    p2d_hypos.append(torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1)) # N x 16 x 2 x (1+k)
                p2d_hypos       = torch.stack(p2d_hypos, dim=0)                 # n_hypos, N, 16, 2, 1+k
                odefunc_ags     = partial(odefunc, p2d=p2d_hypos[...,torch.randperm(c.args.topk+1)])
            else:
                p2d_hypos   = torch.stack([p2d_k[..., torch.randperm(c.args.topk)] for _ in range(c.n_hypo)])
                odefunc_ags = partial(odefunc, p2d=p2d_hypos)
            solver          = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
            x1_all          = solver.sample(time_grid=T, x_init=x0_all.view(-1,17,3), method='midpoint', step_size=c.step_size, return_intermediates=False)
            x1_all          = x1_all.detach().cpu()
            x1_all_ra       = (x1_all - x1_all[:,0:1,:]).view(c.n_hypo,N,J,C)*1e3 # [m] -> [mm]
            p3d_gt_all      = p3d_gt.unsqueeze(0).repeat(c.n_hypo,1,1,1)
            
            errors_proto1   = mpjpe(x1_all_ra, p3d_gt_all)
            errors_proto2   = mpjpe_pa(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1)
            errors_pck      = pck(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1)
            total_err_best_p1.append(torch.min(errors_proto1, dim=0).values) # best hypos
            total_err_best_p2.append(torch.min(errors_proto2, dim=0).values)
            total_pck_best.append(torch.max(errors_pck, dim=0).values)
            total_err_mean_p1.append(torch.mean(errors_proto1, dim=0)) # mean hypos
            total_err_mean_p2.append(torch.mean(errors_proto2, dim=0))
            total_pck_mean.append(torch.mean(errors_pck, dim=0))
            total_err_median_p1.append(torch.median(errors_proto1, dim=0).values) # median hypos
            total_err_median_p2.append(torch.median(errors_proto2, dim=0).values)
            total_pck_median.append(torch.median(errors_pck, dim=0).values)
            total_err_worst_p1.append(torch.max(errors_proto1, dim=0).values) # worst hypos
            total_err_worst_p2.append(torch.max(errors_proto2, dim=0).values)
            total_pck_worst.append(torch.min(errors_pck, dim=0).values)
            bar()
    
    "Stack up the metrics"
    total_err_zero_p1   = torch.cat(total_err_zero_p1)
    total_err_zero_p2   = torch.cat(total_err_zero_p2)
    total_pck_zero      = torch.cat(total_pck_zero)
    total_err_best_p1   = torch.cat(total_err_best_p1)
    total_err_best_p2   = torch.cat(total_err_best_p2)
    total_pck_best      = torch.cat(total_pck_best)
    total_err_mean_p1   = torch.cat(total_err_mean_p1)
    total_err_mean_p2   = torch.cat(total_err_mean_p2)
    total_pck_mean      = torch.cat(total_pck_mean)
    total_err_median_p1 = torch.cat(total_err_median_p1)
    total_err_median_p2 = torch.cat(total_err_median_p2)
    total_pck_median    = torch.cat(total_pck_median)
    total_err_worst_p1  = torch.cat(total_err_worst_p1)
    total_err_worst_p2  = torch.cat(total_err_worst_p2)
    total_pck_worst     = torch.cat(total_pck_worst)
    
    "Averaging all metrics"
    mpjpe_zero          = round(torch.mean(total_err_zero_p1).item(),1)
    mpjpe_pa_zero       = round(torch.mean(total_err_zero_p2).item(),1)
    pck_zero            = round(torch.mean(total_pck_zero).item(),1)
    mpjpe_best          = round(torch.mean(total_err_best_p1).item(),1)
    mpjpe_pa_best       = round(torch.mean(total_err_best_p2).item(),1)
    pck_best            = round(torch.mean(total_pck_best).item(),1)
    mpjpe_mean          = round(torch.mean(total_err_mean_p1).item(),1)
    mpjpe_pa_mean       = round(torch.mean(total_err_mean_p2).item(),1)
    pck_mean            = round(torch.mean(total_pck_mean).item(),1)
    mpjpe_median        = round(torch.mean(total_err_median_p1).item(),1)
    mpjpe_pa_median     = round(torch.mean(total_err_median_p2).item(),1)
    pck_median          = round(torch.mean(total_pck_median).item(),1)
    mpjpe_worst         = round(torch.mean(total_err_worst_p1).item(),1)
    mpjpe_pa_worst      = round(torch.mean(total_err_worst_p2).item(),1)
    pck_worst           = round(torch.mean(total_pck_worst).item(),1)
    
    "Show results in terminal"
    print("\nAverage results:")
    table               = PrettyTable()
    table.field_names   = ["Type", "MPJPE\u2193", "MPJPE-PA\u2193", "PCK\u2191"]
    table.add_row(["Zero",  mpjpe_zero,   mpjpe_pa_zero,   pck_zero])
    table.add_row(["Best",  mpjpe_best,   mpjpe_pa_best,   pck_best])
    table.add_row(["Mean",  mpjpe_mean,   mpjpe_pa_mean,   pck_mean])
    table.add_row(["Media", mpjpe_median, mpjpe_pa_median, pck_median])
    table.add_row(["Worst", mpjpe_worst,  mpjpe_pa_worst,  pck_worst])
    table.align["Type"] = "l"
    print(table)
    print()

    if c.args.wandb:
        wandb.log({'P1_'+mode: mpjpe_best})
        wandb.log({'P2_'+mode: mpjpe_pa_best})
        wandb.log({'PCK_'+mode: pck_best})

if __name__ == "__main__":

    print("Program is running on: ", c.device)
    print("Running experiment: ", c.args.exp, "\n")
    seed_everything(c.args.seed)
    testset, GSset, noGSset, outdoorset = create_loaders_3dhp(dataset, 'org') # 3DHP from original HRNet
    odefunc = FlowNet(hid_dim=1024, act_fn=c.args.act, k=c.args.topk, n_layers=c.args.num_gcn).to(c.device)
    print("FlowNet num. params: ", sum(p.numel() for p in odefunc.parameters()))
    
    if c.args.wandb:
        config = {
            "exp": c.args.exp,
            "seed": c.args.seed,
            "activation": c.args.act,
            "topk": c.args.topk,
            "n_gcn": c.args.num_gcn
        }
        wandb.init(
            project="FMPose_" + c.args.exp,
            name="seed_" + str(c.args.seed),
            config=config,
            dir=os.path.join(c.args.dir, 'wandb')
        )
    
    test('GS')      # green screen
    test('noGS')    # no green screen
    test('out')     # outdoor
    test('all')     # all
    