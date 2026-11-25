"""固定既有 CNN3D 轨迹的 UltraG-Ray 凸阵试验；不优化位姿。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .data import write_json
from .losses import masked_ssim
from .trajectory_reconstruction import Inputs, TrajectoryData, WORKSPACE

BASELINE = WORKSPACE / 'logs/20261117_neuf_cnn3d_knn'
INR = WORKSPACE / 'logs/20261117_neuf_cnn3d_800k/fixed/checkpoints/step_800000.pt'
FAN = WORKSPACE / 'UltraNeRF-Studio/data/cerebral/patient_0_convex/dataset_manifest.json'


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ConvexGeometry:
    """原图像素和声束坐标间的显式变换；相机原点移动仅为坐标换基。"""
    def __init__(self, data):
        fan = json.loads(FAN.read_text())['fan_geometry']
        self.cx, self.cy = fan['center_x_px'], fan['center_y_px']
        h, w = data.images.shape[-2:]
        device = data.images.device
        row, col = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        self.radius = torch.hypot(col-self.cx, row-self.cy)
        self.theta = torch.atan2(col-self.cx, row-self.cy)
        union = data.mask.bool().any(0) | data.guidance.reference_mask
        # 扩展既有拟合扇面至实际掩膜全集，避免因拟合边界裁掉有效输入。
        self.rmin = math.floor(float(self.radius[union].min()))-2
        self.rmax = math.ceil(float(self.radius[union].max()))+2
        self.angle = 2*math.degrees(float(self.theta[union].abs().max()))+.5
        self.height = math.ceil(self.rmax-self.rmin)+1
        self.width = math.ceil(self.rmax*math.radians(self.angle))
        self.pitch = float(data.spacing[0])
        self.sag_pitch = float(data.metadata['reference_spacing_mm'])
        basis = torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]], device=device)
        offset = torch.tensor([(self.cy-353.5)*self.pitch,(self.cx-472)*self.pitch,0.], device=device)
        self.c2w = data.initial.clone()
        self.c2w[:,:3,:3] = data.initial[:,:3,:3] @ basis
        self.c2w[:,:3,3] = (data.initial[:,:3,:3] @ offset) + data.initial[:,:3,3]
        self.c2w[:,:3,3] /= 10  # 毫米转厘米，与官方声学积分单位保持一致。
        self.sag_c2w = torch.eye(4,device=device)[None]
        self.sag_c2w[0,:3,:3] = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]],device=device)
        self.sag_c2w[0,:3,3] = torch.tensor([self.cx*self.sag_pitch,self.cy*self.sag_pitch,0.],device=device)/10
        # 在完整像素域验证换基；比较真实世界坐标而非矩阵元素。
        pixels = torch.tensor([[0.,0.],[472.,353.5],[943.,707.],[328.,214.]],device=device)
        old_local = torch.stack(((pixels[:,1]-353.5)*self.pitch,(pixels[:,0]-472)*self.pitch,torch.zeros(4,device=device)),-1)
        cam_local = torch.stack(((pixels[:,0]-self.cx)*self.pitch,torch.zeros(4,device=device),(pixels[:,1]-self.cy)*self.pitch),-1)/10
        old = old_local @ data.initial[:,:3,:3].transpose(1,2) + data.initial[:,None,:3,3]
        new = 10*(cam_local @ self.c2w[:,:3,:3].transpose(1,2) + self.c2w[:,None,:3,3])
        self.world_error_mm = float((old-new).abs().max())
        assert self.world_error_mm < 5e-5, self.world_error_mm
        self.native_shape = (h,w)

    def config(self, factor=1, sag=False):
        h = math.ceil(self.height/factor)
        w = math.ceil(self.width/factor)
        pitch = self.sag_pitch if sag else self.pitch
        dr = (self.rmax-self.rmin)/(h-1)
        # CUDA使用(i+.5)/(H-1)，边界减半采样使首末中心对应rmin/rmax。
        return dict(height=h,width=w,near_plane=(self.rmin-.5*dr)*pitch/10,
                    far_plane=(self.rmax-.5*dr)*pitch/10,opening_angle=self.angle)

    def to_polar(self, images, factor=1):
        cfg=self.config(factor)
        radius=torch.linspace(self.rmin,self.rmax,cfg['height'],device=images.device)
        theta=(torch.arange(cfg['width'],device=images.device)+.5)/cfg['width']*math.radians(self.angle)-math.radians(self.angle)/2
        x=self.cx+radius[:,None]*theta.sin()[None]
        y=self.cy+radius[:,None]*theta.cos()[None]
        h,w=self.native_shape
        grid=torch.stack((x/(w-1)*2-1,y/(h-1)*2-1),-1)
        return F.grid_sample(images[:,None],grid[None].expand(len(images),-1,-1,-1),align_corners=True)[:,0]

    def to_native(self, polar):
        h,w=polar.shape[-2:]
        xp=(self.theta/math.radians(self.angle)+.5)*w-.5
        yp=(self.radius-self.rmin)/(self.rmax-self.rmin)*(h-1)
        grid=torch.stack((xp/(w-1)*2-1,yp/(h-1)*2-1),-1)
        return F.grid_sample(polar[:,None],grid[None].expand(len(polar),-1,-1,-1),align_corners=True)[:,0]


class GaussianField(nn.Module):
    def __init__(self, data, count=200000):
        super().__init__()
        device=data.images.device
        frames=torch.randint(len(data.images),(count,),device=device)
        # 初始化来自实际采样支持域；不使用KNN/INR的预测灰度作监督。
        flat=torch.empty(count,dtype=torch.long,device=device)
        for frame in range(len(data.images)):
            pick=frames==frame
            valid=torch.as_tensor(data.valid_centers[frame],device=device,dtype=torch.long)
            flat[pick]=valid[torch.randint(len(valid),(int(pick.sum()),),device=device)]
        local=data.local[flat//data.width,flat%data.width]
        xyz=(data.initial[frames,:3,:3]@local[:,:,None])[:,:,0]+data.initial[frames,:3,3]
        xyz=xyz/10+torch.randn_like(xyz)*.015
        quats=torch.zeros(count,4,device=device); quats[:,0]=1
        sh0=torch.full((count,1,1),(.1-.5)/.28209479177387814,device=device)
        self.splats=nn.ParameterDict(dict(means=nn.Parameter(xyz),quats=nn.Parameter(quats),
            scales=nn.Parameter(torch.full((count,3),math.log(.05),device=device)),
            transmittances=nn.Parameter(torch.full((count,),math.log(99.),device=device)),
            sh0=nn.Parameter(sh0),shN=nn.Parameter(torch.zeros(count,3,1,device=device))))

    @classmethod
    def from_checkpoint(cls,checkpoint):
        model=cls.__new__(cls);nn.Module.__init__(model)
        model.splats=nn.ParameterDict({k:nn.Parameter(v) for k,v in checkpoint['splats'].items()})
        return model

    def render(self,c2w,config,degree=1):
        from gsplat.cuda._wrapper import (isect_tiles_rectangular,isect_offset_encode_rectangular,
                                          ultrasound_rasterize_to_pixels,spherical_harmonics)
        p=self.splats
        means,scales=p['means'],p['scales'].exp()
        views=torch.linalg.inv(c2w)
        h,w=config['height'],config['width'];count=len(c2w)
        # 官方投影按tan(theta)分桶，但凸阵CUDA按等角度采样。用保守球包围盒
        # 在实际声束坐标分桶，防止外侧声束漏高斯；成像/反传仍用官方CUDA。
        with torch.no_grad():
            cam=torch.einsum('cij,nj->cni',views[:,:3,:3],means)+views[:,None,:3,3]
            radius=4*scales.amax(-1)[None]
            rho=torch.hypot(cam[...,0],cam[...,2]).clamp_min(1e-7)
            angle=torch.atan2(cam[...,0],cam[...,2])
            half=torch.asin((radius/rho).clamp(max=1))
            fov=math.radians(config['opening_angle'])
            valid=(cam[...,1].abs()<=radius)&(rho-radius<=config['far_plane'])&(cam[...,2]+radius>0)
            centers=torch.stack(((angle/fov+.5)*w,torch.full_like(angle,h/2)),-1).contiguous()
            rx=torch.ceil(half/fov*w+1).to(torch.int32)
            radii=torch.stack((rx,torch.full_like(rx,h*2)),-1)
            radii=torch.where(valid[...,None],radii,0).contiguous()
            depths=rho.contiguous()
            tx,ty=4,128;tw,th=math.ceil(w/tx),math.ceil(h/ty)
            _,ids,flatten=isect_tiles_rectangular(centers,radii,depths,tx,ty,tw,th,packed=False,n_images=count)
            offsets=isect_offset_encode_rectangular(ids,count,tw,th).reshape(count,th,tw)
        sh=torch.cat((p['sh0'],p['shN']),1)[None].expand(count,-1,-1,-1)
        dirs=means[None]-c2w[:,None,:3,3]
        intensities=torch.clamp_min(spherical_harmonics(degree,dirs,sh,masks=valid)+1,.5)
        trans=p['transmittances'].sigmoid()[None].expand(count,-1)
        result=ultrasound_rasterize_to_pixels(means,p['quats'],scales,intensities,trans,views,
            w,h,tx,ty,offsets,flatten,convex=True,near_plane=config['near_plane'],
            far_plane=config['far_plane'],opening_angle=config['opening_angle'],opening_width=0.)
        return (*result,dict(radii=radii))


def validate_renderer():
    """三个已知高斯检查等角声束、前向值及位置梯度；不训练模型。"""
    model=GaussianField.__new__(GaussianField); nn.Module.__init__(model)
    angle=torch.tensor([-25.,0.,25.],device='cuda')*math.pi/180
    means=torch.stack((10*angle.sin(),torch.zeros_like(angle),10*angle.cos()),-1)
    quat=torch.zeros(3,4,device='cuda');quat[:,0]=1
    model.splats=nn.ParameterDict(dict(means=nn.Parameter(means),quats=nn.Parameter(quat),
        scales=nn.Parameter(torch.full((3,3),math.log(.08),device='cuda')),
        transmittances=nn.Parameter(torch.full((3,),16.,device='cuda')),
        sh0=nn.Parameter(torch.full((3,1,1),(.8-1)/.28209479177387814,device='cuda')),
        shN=nn.Parameter(torch.zeros(3,3,1,device='cuda'))))
    cfg=dict(width=96,height=65,near_plane=8.,far_plane=12.,opening_angle=70.)
    pose=torch.eye(4,device='cuda')[None]
    image,*_=model.render(pose,cfg,0)
    theta=(torch.arange(96,device='cuda')+.5)/96*math.radians(70)-math.radians(35)
    radius=8+4*(torch.arange(65,device='cuda')+.5)/64
    points=torch.stack((radius[:,None]*theta.sin()[None],torch.zeros(65,96,device='cuda'),
                        radius[:,None]*theta.cos()[None]),-1)
    alpha=torch.exp(-.5*((points[:,:,None]-means)/.08).square().sum(-1)).clamp(max=.999)
    expected=.8*(1-torch.exp(-alpha.sum(-1)))
    error=float((image[0,:,:,0]-expected).abs().max())
    assert error<.004,error
    weights=torch.linspace(.1,1.,96,device='cuda')[None,None,:,None]
    loss=(image*weights).mean();loss.backward()
    analytic=float(model.splats['means'].grad[0,0])
    assert math.isfinite(analytic) and abs(analytic)>1e-8
    with torch.no_grad():
        eps=.001;model.splats['means'][0,0]+=eps
        plus=float((model.render(pose,cfg,0)[0]*weights).mean())
        model.splats['means'][0,0]-=2*eps
        minus=float((model.render(pose,cfg,0)[0]*weights).mean())
    numerical=(plus-minus)/(2*eps)
    assert abs(analytic-numerical)<max(1e-5,.1*abs(numerical)),(analytic,numerical)
    return dict(forward_max_error=error,position_gradient=analytic,finite_difference_gradient=numerical)


def initialize(output, steps):
    torch.set_num_threads(4)
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    data=TrajectoryData(Inputs('cuda'),'CNN3D')
    ck=torch.load(INR,map_location='cpu',weights_only=False)
    assert ck['completed_steps']==800000 and ck['pose_updates']==0
    assert torch.equal(ck['corrected_poses'],data.initial.cpu())
    assert ck['data_metadata']['mask_sha256']==data.metadata['mask_sha256']
    del ck
    geometry=ConvexGeometry(data)
    config=dict(method='UltraG-Ray convex adaptation',steps=steps,seed=3407,initial_gaussians=200000,
        max_gaussians=300000,batch_size=1,train_resolution_factor=2,pose_optimization=False,pose_jitter=False,
        image_loss='masked 1-SSIM; no MSE',scale_regularization=.001,training_frames=list(range(242)),
        milestones=sorted(set([math.ceil(steps/2),math.ceil(3*steps/4),steps])),
        data=data.metadata,pose_tensor_sha256=tensor_hash(data.initial),baseline=str(BASELINE),inr=str(INR),
        gaussian_source_commit=subprocess.check_output(['git','-C',str(WORKSPACE/'UltraG-Ray'),'rev-parse','HEAD'],text=True).strip(),
        runtime=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            cuda_home=os.environ.get('CUDA_HOME'),job_id=os.environ.get('PBS_JOBID'),
            nvcc=subprocess.check_output(['nvcc','--version'],text=True).strip()),
        fan_source=str(FAN),fan_center_xy_px=[geometry.cx,geometry.cy],geometry_world_max_error_mm=geometry.world_error_mm,
        native_polar_shape_hw=[geometry.height,geometry.width],train_probe=geometry.config(2),
        renderer_adaptation='official ultrasound forward/backward CUDA; conservative equal-angle spherical bounds replace pinhole tile projection',
        comparison='same 242 input frames and frozen poses; historical INR uses MSE/800k, Gaussian SSIM/6k; not an equal-budget loss-controlled ablation',
        validation='COR fitting fidelity; SAG informed upstream trajectory; no independent held-out 3D truth')
    for sub in ['checkpoints','metrics','plots','predictions','run_config/source']:
        (output/sub).mkdir(parents=True,exist_ok=True)
    config_path=output/'run_config/config.json'
    source_dir=output/'run_config/source'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        for key in ['steps','seed','pose_tensor_sha256','initial_gaussians','train_probe']:
            assert config[key]==previous[key],key
        session_id=os.environ.get('PBS_JOBID','manual').split('.')[0]
        write_json(output/'run_config'/f'session_{session_id}.json',dict(runtime=config['runtime'],original_runtime=previous['runtime']))
        config=previous
        source_dir=output/'run_config'/f'source_session_{session_id}'
        source_dir.mkdir(exist_ok=True)
    else:
        write_json(config_path,config)
        write_json(output/'run_config/input_metadata.json',data.metadata)
    for name in ['gaussian_reconstruction.py','gaussian_evaluation.py']:
        shutil.copy2(Path(__file__).with_name(name),source_dir/name)
    shutil.copy2(WORKSPACE/'qsub/neuf/gaussian.sh',source_dir/'gaussian.sh')
    aggregate=output.parent.parent/'index_all/UltraGRay'
    (aggregate/'metrics').mkdir(parents=True,exist_ok=True)
    write_json(aggregate/'run_config/comparison_manifest.json',dict(comparison_indices=data.comparison,
        roi_ids=data.guidance.roi if isinstance(data.guidance.roi,list) else ['R1','R2','R3','R4','R5'],
        normalization='uint8 /255, display 0..1, no method-specific rescaling',pose_sha256=config['pose_tensor_sha256']))
    return data,geometry,config,aggregate


def save_checkpoint(output,model,optimizers,schedulers,strategy_state,data,config,step):
    path=output/'checkpoints'/f'step_{step:06d}.pt'
    assert not path.exists(), path
    torch.save(dict(splats=model.splats.state_dict(),completed_steps=step,total_steps=config['steps'],config=config,
        poses=data.initial.cpu(),optimizers={k:v.state_dict() for k,v in optimizers.items()},
        schedulers={k:v.state_dict() for k,v in schedulers.items()},strategy_state=strategy_state,
        random_state=random.getstate(),numpy_state=np.random.get_state(),torch_state=torch.get_rng_state(),
        cuda_state=torch.cuda.get_rng_state_all()),path)
    return path


def train(args):
    from gsplat.strategy.ultrasound import Ultrasound3DStrategy
    from .gaussian_evaluation import evaluate
    output=args.output
    renderer_audit=validate_renderer()
    data,geometry,config,aggregate=initialize(output,args.steps)
    write_json(output/'metrics/renderer_audit.json',renderer_audit)
    checkpoint=None
    if args.resume is not None:
        assert args.resume.resolve().parent==(output/'checkpoints').resolve()
        checkpoint=torch.load(args.resume,map_location='cuda',weights_only=False)
        assert torch.equal(checkpoint['poses'].cpu(),data.initial.cpu())
        assert checkpoint['total_steps']==args.steps
    model=GaussianField.from_checkpoint(checkpoint) if checkpoint else GaussianField(data,config['initial_gaussians'])
    scale=float(torch.linalg.norm(data.bounds[1]-data.bounds[0]))/20
    lr=dict(means=1e-4*scale,quats=5e-3,scales=5e-3,transmittances=5e-4,sh0=5e-3,shN=1e-5)
    optimizers={key:torch.optim.Adam([value],lr=lr[key],eps=1e-15) for key,value in model.splats.items()}
    schedulers={k:torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.1**(1/args.steps)) for k,opt in optimizers.items()}
    strategy=Ultrasound3DStrategy(max_gaussians=config['max_gaussians'],refine_start_iter=500,
        refine_stop_iter=int(args.steps*2/3),refine_every=500,verbose=True)
    state=strategy.initialize_state(scene_scale=scale)
    start_step=0
    if checkpoint:
        start_step=checkpoint['completed_steps']
        for key,opt in optimizers.items():opt.load_state_dict(checkpoint['optimizers'][key])
        for key,sched in schedulers.items():sched.load_state_dict(checkpoint['schedulers'][key])
        state=checkpoint['strategy_state']
        random.setstate(checkpoint['random_state']);np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_state']])
    targets=geometry.to_polar(data.images,2)
    masks=geometry.to_polar(data.mask,2)>.99999
    assert masks.sum(dim=(1,2)).min()>1000
    write_json(output/'metrics/geometry_audit.json',dict(world_coordinate_max_error_mm=geometry.world_error_mm,
        poses_identical_to_inr=True,training_mask_pixels=[int(x) for x in masks.sum((1,2))],
        note='reuse historical image-derived fan center; not an independently measured acoustic calibration'))
    torch.cuda.synchronize()
    start=time.perf_counter(); started=datetime.now(timezone.utc).isoformat(); completed=start_step; final_path=args.resume; step_seconds=0.
    previous_wall=0.
    if checkpoint:
        with (aggregate/'metrics/timing.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert int(rows[-1]['global_step'])==start_step
        step_seconds=sum(float(row['step_time_sec']) for row in rows)
        previous_summary=json.loads((aggregate/'metrics/training_summary.json').read_text())
        started=previous_summary['start_time'];previous_wall=previous_summary['total_wall_seconds']
    status='failed'
    with (aggregate/'metrics/timing.csv').open('a' if checkpoint else 'w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','global_step','step_time_sec','elapsed_sec','loss','ssim','gaussians'])
        if not checkpoint:writer.writeheader()
        try:
            if checkpoint:
                evaluate(output,model,data,geometry,start_step,full=start_step==args.steps)
            bar=tqdm(range(start_step+1,args.steps+1),total=args.steps,initial=start_step,mininterval=15,desc='Fixed-pose UltraG-Ray')
            for step in bar:
                tic=time.perf_counter()
                frame=torch.randint(242,(1,),device='cuda')
                degree=0 if step<500 else 1
                pred,_,_,_,info=model.render(geometry.c2w[frame],geometry.config(2),degree)
                pred=pred.permute(0,3,1,2)
                score,_=masked_ssim(pred,targets[frame,None],masks[frame,None])
                loss=1-score+config['scale_regularization']*model.splats['scales'].exp().mean()
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward()
                strategy.step_post_backward(model.splats,optimizers,state,step,info)
                for opt in optimizers.values(): opt.step(); opt.zero_grad(set_to_none=True)
                for sched in schedulers.values(): sched.step()
                torch.cuda.synchronize(); completed=step
                duration_step=time.perf_counter()-tic;step_seconds+=duration_step
                writer.writerow(dict(epoch=step/242,global_step=step,step_time_sec=duration_step,
                    elapsed_sec=previous_wall+time.perf_counter()-start,loss=float(loss),ssim=float(score),gaussians=len(model.splats['means'])))
                if step%25==0:
                    stream.flush(); bar.set_postfix(loss=f'{float(loss):.4f}',ssim=f'{float(score):.4f}',G=len(model.splats['means']),refresh=False)
                if step in config['milestones']:
                    assert tensor_hash(data.initial)==config['pose_tensor_sha256']
                    final_path=save_checkpoint(output,model,optimizers,schedulers,state,data,config,step)
                    evaluate(output,model,data,geometry,step,full=step==args.steps)
                    print('MILESTONE',step,final_path,flush=True)
            status='completed'
        finally:
            duration=time.perf_counter()-start
            write_json(aggregate/'metrics/training_summary.json',dict(status=status,configured_steps=args.steps,
                completed_steps=completed,configured_epochs=args.steps/242,completed_epochs=completed/242,
                start_time=started,end_time=datetime.now(timezone.utc).isoformat(),total_wall_seconds=previous_wall+duration,
                session_start_step=start_step,session_wall_seconds=duration,
                mean_step_seconds=step_seconds/max(completed,1),final_checkpoint=str(final_path)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','audit','evaluate','compare-checkpoints'],required=True)
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    assert args.output.resolve().is_relative_to(WORKSPACE/'logs')
    if args.mode=='train': train(args)
    elif args.mode=='audit':
        data,geom,config,_=initialize(args.output,args.steps)
        model=GaussianField(data,1000)
        pred,*_=model.render(geom.c2w[:1],geom.config(4),0)
        loss=pred.mean();loss.backward()
        assert model.splats['means'].grad is not None and torch.isfinite(model.splats['means'].grad).all()
        print('GEOMETRY_AND_GRADIENT_OK',config['geometry_world_max_error_mm'],pred.shape,flush=True)
    elif args.mode=='compare-checkpoints':
        from .gaussian_evaluation import render_cor_examples
        data,geom,config,_=initialize(args.output,args.steps)
        records=[]
        for step in config['milestones']:
            path=args.output/'checkpoints'/f'step_{step:06d}.pt'
            ck=torch.load(path,map_location='cuda',weights_only=False)
            assert ck['completed_steps']==step and ck['total_steps']==args.steps
            assert torch.equal(data.initial,ck['poses'].to('cuda'))
            assert all(k in ck for k in ['optimizers','schedulers','random_state','numpy_state','torch_state','cuda_state'])
            assert all(torch.isfinite(v).all() for v in ck['splats'].values())
            model=GaussianField.from_checkpoint(ck)
            render_cor_examples(args.output,model,data,geom,step)
            records.append(dict(step=step,path=str(path),bytes=path.stat().st_size,poses_exact=True,states_present=True))
            print('CHECKPOINT_VERIFIED',step,flush=True)
            del model,ck
        write_json(args.output/'metrics/checkpoint_verification.json',records)
    else:
        from .gaussian_evaluation import evaluate
        data,geom,config,_=initialize(args.output,args.steps)
        ck=torch.load(args.output/'checkpoints'/f'step_{args.steps:06d}.pt',map_location='cuda',weights_only=False)
        model=GaussianField.from_checkpoint(ck)
        assert torch.equal(data.initial,ck['poses'].to('cuda'))
        evaluate(args.output,model,data,geom,args.steps,full=True)


if __name__=='__main__': main()
