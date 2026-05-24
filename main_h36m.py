# Cuong Le - CVL, Linköping University
import os
import math
import numpy as np
import torch

# database
import data.data_h36m as dataset
import config as c
from utils.data import *
from utils.metrics import *
from models.model import FlowNet, WrappedModel
from alive_progress import alive_bar
from sklearn.metrics import auc

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from functools import partial

# statistic
import wandb
from prettytable import PrettyTable

def train():
    print('--- Setting up training ---')
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True) # drop last smaller batch
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False)
    optimizer       = torch.optim.AdamW(odefunc.parameters(), lr=c.lr)
    scheduler       = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.lr_step, gamma=0.1)
    path            = AffineProbPath(scheduler=CondOTScheduler()) # Instantiating an affine OT path
    loss_fn         = torch.nn.MSELoss(reduction='mean') # L2 loss
    best_epoch, best_p1 = 1, 100 # arbitrary
    print('--- Training started! ---')
    print()
    for epoch in range(1,c.n_epochs+1):
        
        "--Training--" 
        print_log('\t-- Epoch {:d}, LR: {:.2e} --'.format(epoch, optimizer.param_groups[0]['lr']))
        odefunc.train()
        total_iters = math.ceil(len(train_dataset)/c.batch_size)-1
        total_loss  = []
        with alive_bar(total=total_iters, title='Training', length=15, bar='circles', spinner='flowers', dual_line=True, refresh_secs=0.3) as bar:
            for _, batch in enumerate(train_loader):
                p2d_k   = batch["p2d_topk"].to(c.device)                    # N x 16 x 2 x k
                p2d_max = batch["p2d"].to(c.device)                         # N x 16 x 2
                p3d     = batch["p3d"].to(c.device)                         # N x 17 x 3
                t       = torch.rand(p3d.shape[0]).to(c.device)             # N x 1
                x0, x1  = c.std_dev * torch.randn_like(p3d), p3d.clone()    # N x 17 x 3
                path_t  = path.sample(t=t.float(), x_0=x0, x_1=x1)
                if c.args.sampling:
                    k_indices = torch.randperm(96)[:c.args.topk]
                    p2d     = torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1) # N x 16 x 2 x (1+k)
                    dx_t    = odefunc(path_t.x_t, path_t.t, p2d[...,torch.randperm(c.args.topk+1)]) # N x 1, N x 17 x 3
                else:
                    dx_t    = odefunc(path_t.x_t, path_t.t, p2d_k[...,torch.randperm(c.args.topk)]) # N x 1, N x 17 x 3
                loss    = loss_fn(dx_t, path_t.dx_t)
                optimizer.zero_grad()                                       # clear grad
                loss.backward()                                             # backward
                optimizer.step()                                            # update
                total_loss.append(loss.item())
                bar.text = ('Loss: {:.4f}'.format(loss.item()))
                bar()
        if c.args.wandb: wandb.log({'epoch': epoch, 'train_loss': np.mean(total_loss)})
        print_log('\tTotal loss: {:.4f}'.format(np.mean(total_loss)))
        scheduler.step()
        print('------------------------------')
        
        
        "--Validating--"
        if not (epoch % 2): # Evaluation at every 2nd training step
            print_log('Validating')
            odefunc.eval()
            total_iters         = math.ceil(len(val_dataset)/(c.batch_size))
            total_err_best_p1   = []
            with alive_bar(total=total_iters, title='Testing', length=10, bar='bubbles', spinner='elements', refresh_secs=0.3) as bar:
                for _, batch in enumerate(val_loader):
                    p2d_k       = batch["p2d_topk"].to(c.device)                # N x 16 x 2 x k
                    p2d_max     = batch["p2d"].to(c.device)                     # N x 16 x 2
                    p3d         = batch["p3d"].to(c.device)                     # N x 17 x 3
                    p3d_gt      = batch["p3d_gt"]                               # root-aligned at origin [mm]
                    N, J, C     = p3d.shape
                    
                    x0_all      = c.std_dev * torch.randn_like(p3d.unsqueeze(0).repeat(c.n_hypo,1,1,1))
                    if c.args.sampling:
                        p2d_hypos   = []
                        for _ in range(c.n_hypo):
                            k_indices   = torch.randperm(96)[:c.args.topk]
                            p2d_hypos.append(torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1)) # N x 16 x 2 x (1+k)
                        p2d_hypos   = torch.stack(p2d_hypos, dim=0)                 # n_hypos, N, 16, 2, 1+k
                        odefunc_ags = partial(odefunc, p2d=p2d_hypos[...,torch.randperm(c.args.topk+1)])
                    else:
                        p2d_hypos   = torch.stack([p2d_k[...,torch.randperm(c.args.topk)] for _ in range(c.n_hypo)])
                        odefunc_ags = partial(odefunc, p2d=p2d_hypos)
                    solver      = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
                    T           = torch.linspace(0,1,c.n_steps).to(c.device)    # sampling steps
                    x1_all      = solver.sample(time_grid=T, x_init=x0_all.view(-1,17,3), method='midpoint', step_size=c.step_size, return_intermediates=False)
                    x1_all      = x1_all.detach().cpu()                         # detach to lower memory
                    x1_all_ra   = (x1_all - x1_all[:,0:1,:]).view(c.n_hypo,N,J,C)*1e3 # [m] -> [mm]
                    p3d_gt_all  = p3d_gt.unsqueeze(0).repeat(c.n_hypo,1,1,1)    
                    
                    errors_proto1   = mpjpe(x1_all_ra, p3d_gt_all)
                    errors_pck      = pck(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1)
                    total_err_best_p1.append(torch.min(errors_proto1, dim=0).values) # best hypothesis
                    bar()
                    
            total_err_best_p1   = torch.cat(total_err_best_p1)
            mpjpe_best          = round(torch.mean(total_err_best_p1).item(),1)
            print("3D Protocol-I best hypo: %.1f \n" % mpjpe_best)
            if c.args.wandb: wandb.log({'epoch': epoch, 'val_best': mpjpe_best})
            if (epoch > c.lr_step):
                if (mpjpe_best < best_p1):
                    best_epoch, best_p1  = epoch, mpjpe_best
                    print('epoch: ' + str(best_epoch) + '\n')
                    odefunc.save(c.model_best)
    print('--- Training done! ---')
    print('--- Best model is at epoch {:d} with MPJPE of {:.1f}'.format(best_epoch, best_p1))
    odefunc.save(c.model_last)

def test(hard: bool=False):
    if not hard:
        dataset         = easy_dataset
        print("-- on EASY dataset -- ")
    else:
        dataset         = hard_dataset
        print("-- on HARD dataset -- ")
    
    test_loader         = torch.utils.data.DataLoader(dataset, batch_size=c.batch_size, shuffle=False, drop_last=False)
    odefunc.load(c.model_best, c.device)
    odefunc.eval()
    
    total_iters         = math.ceil(len(dataset)/c.batch_size)
    total_err_zero_p1, total_err_zero_p2, total_pck_zero        = [], [], []
    total_err_mean_p1, total_err_mean_p2, total_pck_mean        = [], [], []
    total_err_worst_p1, total_err_worst_p2, total_pck_worst     = [], [], []
    total_err_best_p1, total_err_best_p2, total_pck_best        = [], [], []
    total_err_median_p1, total_err_median_p2, total_pck_median  = [], [], []
    total_cps_best = []
    
    with alive_bar(total=total_iters, title='Testing', length=10, bar='bubbles', spinner='elements', refresh_secs=0.3) as bar:
        for _, batch in enumerate(test_loader):
            p2d_k       = batch["p2d_topk"].to(c.device)                # N x 16 x 2 x k
            p2d_max     = batch["p2d"].to(c.device)                     # N x 16 x 2
            p3d         = batch["p3d"].to(c.device)
            p3d_gt      = batch["p3d_gt"]
            N, J, C     = p3d.shape
            
            T           = torch.linspace(0,1,c.n_steps).to(c.device)  # sample times
            x0          = torch.zeros_like(p3d)
            if c.args.sampling:
                k_indices   = torch.randperm(96)[:c.args.topk]
                p2d         = torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1) # N x 16 x 2 x (1+k)
                odefunc_ags = partial(odefunc, p2d=p2d[...,torch.randperm(c.args.topk+1)])
            else:
                odefunc_ags = partial(odefunc, p2d=p2d_k[...,torch.randperm(c.args.topk)])
            solver      = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
            x1          = solver.sample(time_grid=T, x_init=x0, method='midpoint', step_size=c.step_size, return_intermediates=False)
            x1          = x1.detach().cpu()
            x1_ra       = (x1 - x1[:,0:1,:])*1e3 # [m] -> [mm]
            total_err_zero_p1.append(mpjpe(x1_ra, p3d_gt))
            total_err_zero_p2.append(mpjpe_pa(x1_ra, p3d_gt))
            total_pck_zero.append(pck(x1_ra, p3d_gt))
            
            x0_all      = c.std_dev * torch.randn_like(p3d.unsqueeze(0).repeat(c.n_hypo,1,1,1))
            if c.args.sampling:
                p2d_hypos   = []
                for _ in range(c.n_hypo):
                    k_indices = torch.randperm(96)[:c.args.topk]
                    p2d_hypos.append(torch.cat((p2d_max[...,None], p2d_k[...,k_indices]), dim=-1)) # N x 16 x 2 x (1+k)
                p2d_hypos   = torch.stack(p2d_hypos, dim=0) # n_hypos, N, 16, 2, 1+k
                odefunc_ags = partial(odefunc, p2d=p2d_hypos[...,torch.randperm(c.args.topk+1)])
            else:
                p2d_hypos   = torch.stack([p2d_k[...,torch.randperm(c.args.topk)] for _ in range(c.n_hypo)])
                odefunc_ags = partial(odefunc, p2d=p2d_hypos)
            solver      = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
            x1_all      = solver.sample(time_grid=T, x_init=x0_all.view(-1,17,3), method='midpoint', step_size=c.step_size, return_intermediates=False)
            x1_all      = x1_all.detach().cpu()
            x1_all_ra   = (x1_all - x1_all[:,0:1,:]).view(c.n_hypo,N,J,C)*1e3 # [m] -> [mm]
            p3d_gt_all  = p3d_gt.unsqueeze(0).repeat(c.n_hypo,1,1,1)
            
            errors_proto1   = mpjpe(x1_all_ra, p3d_gt_all)
            errors_proto2   = mpjpe_pa(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1)
            errors_pck      = pck(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1)
            errors_cps      = cps(x1_all_ra, p3d_gt_all).view(c.n_hypo,-1,300)
            
            total_err_best_p1.append(torch.min(errors_proto1, dim=0).values) # best hypos
            total_err_best_p2.append(torch.min(errors_proto2, dim=0).values)
            total_pck_best.append(torch.max(errors_pck, dim=0).values)
            total_cps_best.append(torch.max(errors_cps, dim=0).values)
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
    total_cps_best      = torch.cat(total_cps_best)
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
    auc_cps_best        = round(auc(np.arange(1, 301, 1), total_cps_best.mean(0).cpu().numpy()),1)
    
    "Show results in terminal"
    print("\nAverage results:")
    table               = PrettyTable()
    table.field_names   = ["Type", "MPJPE\u2193", "MPJPE-PA\u2193", "PCK\u2191", "CPS\u2191"]
    table.add_row(["Zero",  mpjpe_zero,   mpjpe_pa_zero,   pck_zero, 0])
    table.add_row(["Best",  mpjpe_best,   mpjpe_pa_best,   pck_best, auc_cps_best])
    table.add_row(["Mean",  mpjpe_mean,   mpjpe_pa_mean,   pck_mean, 0])
    table.add_row(["Media", mpjpe_median, mpjpe_pa_median, pck_median, 0])
    table.add_row(["Worst", mpjpe_worst,  mpjpe_pa_worst,  pck_worst, 0])
    table.align["Type"] = "l"
    print(table)
    print()
    
    if c.args.wandb:
        if not hard:
            wandb.log({'easy_zero_p1': mpjpe_zero})
            wandb.log({'easy_best_p1': mpjpe_best})
            wandb.log({'easy_zero_p2': mpjpe_pa_zero})
            wandb.log({'easy_best_p2': mpjpe_pa_best})
            wandb.log({'easy_zero_pck': pck_zero})
            wandb.log({'easy_best_pck': pck_best})
            wandb.log({'easy_best_cps': auc_cps_best})
        else:
            wandb.log({'hard_zero_p1': mpjpe_zero})
            wandb.log({'hard_best_p1': mpjpe_best})
            wandb.log({'hard_zero_p2': mpjpe_pa_zero})
            wandb.log({'hard_best_p2': mpjpe_pa_best})
            wandb.log({'hard_zero_pck': pck_zero})
            wandb.log({'hard_best_pck': pck_best})
            wandb.log({'hard_best_cps': auc_cps_best})
    
if __name__ == "__main__":

    print("Program is running on: ", c.device)
    print("Running experiment: ", c.args.exp, "\n")
    seed_everything(c.args.seed)
    train_dataset, val_dataset, easy_dataset, hard_dataset = create_loaders_h36m(dataset) # change to train=True to get the train split
    if c.args.sampling: odefunc = FlowNet(hid_dim=1024, act_fn=c.args.act, k=c.args.topk+1, n_layers=c.args.num_gcn).to(c.device)
    else: odefunc = FlowNet(hid_dim=1024, act_fn=c.args.act, k=c.args.topk, n_layers=c.args.num_gcn).to(c.device)
    print("FlowNet num. params: ", sum(p.numel() for p in odefunc.parameters()))
    
    if c.args.wandb:
        config = {
            "exp": c.args.exp,
            "seed": c.args.seed,
            "activation": c.args.act,
            "topk": c.args.topk,
            "n_gcn": c.args.num_gcn,
            "sampling": c.args.sampling,
            "solver": c.args.solver,
            "steps": c.args.n_steps,
        }
        wandb.init(
            project="FMPose_"+c.args.exp,
            name="seed_"+str(c.args.seed),
            config=config,
            save_code=False,
            dir=os.path.join(c.args.dir, 'wandb')
        )
    
    train()
    test()
    test(hard=True)
    
    print("----- Finished experiments on H36M -----")
