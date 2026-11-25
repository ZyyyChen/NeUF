"""在已归档KNN/INR的固定物理坐标和支持域上比较Gaussian。"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from .data import write_json
from .evaluation import _masked_point_metrics
from .losses import masked_ssim
from .trajectory_reconstruction import WORKSPACE


def scores(pred,ref,mask):
    point=_masked_point_metrics(pred,ref,mask)
    t=lambda x:torch.as_tensor(np.ascontiguousarray(x))[None,None]
    score,n=masked_ssim(t(pred),t(ref),t(mask))
    return dict(ssim=float(score),windows=n,psnr=-10*math.log10(max(point['mse'],1e-15)),
                mae=point['mae'],gradient_ncc=point['gradient_ncc'],pixels=int(mask.sum()))


class Baselines:
    def __init__(self,root):
        self.root=root
        self.grid=json.loads((root/'volumes/grid.json').read_text())
        self.volumes={name:np.memmap(root/'volumes'/f'{name}.raw',mode='r',
            dtype=np.uint8 if name=='support' else np.float32,shape=tuple(self.grid['shape_zyx']))
            for name in ['knn','inr','support']}

    def sample(self,xyz):
        coords=np.moveaxis((xyz-np.array(self.grid['origin_xyz_mm']))/self.grid['spacing_mm'],-1,0)[::-1]
        pred={name:ndi.map_coordinates(v,coords,order=1,prefilter=False,output=np.float32)
              for name,v in self.volumes.items()}
        return {k:pred[k] for k in ['knn','inr']},pred['support']>.99999


def fixed_patches(data):
    rng=np.random.default_rng(94117); starts=[]; size=48
    masks=data.mask.bool().cpu().numpy()
    for frame,valid in enumerate(data.valid_centers):
        chosen=[]
        while len(chosen)<2:
            y,x=np.divmod(rng.choice(valid),data.width); r,c=int(y)-size//2,int(x)-size//2
            if r>=0 and c>=0 and r+size<=data.height and c+size<=data.width and masks[frame,r:r+size,c:c+size].all():
                chosen.append((r,c))
        starts.extend(chosen)
    expected='3f0ce85f765bb362dbf55b26698ab843dbb4e834b7aec4e6725c4efe2d7fcc11'
    assert hashlib.sha256(np.asarray(starts,dtype=np.int64).tobytes()).hexdigest()==expected
    return starts


def comparison_plot(output,images,common,truth,roi,step):
    names=['reference','knn','inr','gaussian']
    titles=['Measured SAG','KNN (0.25 mm)','INR (800k)','Gaussian ('+str(step)+')']
    x,y,w,h=roi['roi_xywh'][3]
    fig,axes=plt.subplots(2,4,figsize=(14,6.6),layout='constrained')
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#dedede')
    for col,(name,title) in enumerate(zip(names,titles)):
        value=np.where(common,images[name],np.nan)
        axes[0,col].imshow(value,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        axes[1,col].imshow(value[y:y+h,x:x+w],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        axes[1,col].contour(truth[y:y+h,x:x+w],levels=[.5],colors=['#e5ab3b'],linewidths=.8)
        axes[0,col].set_title(title,fontsize=11)
        for ax in axes[:,col]: ax.set(xticks=[],yticks=[])
    axes[0,0].set_ylabel('Full SAG',fontsize=11)
    axes[1,0].set_ylabel('Fixed R4',fontsize=11)
    fig.suptitle('Same 242 COR frames and fixed CNN3D poses | Gray range 0-1',fontsize=12)
    fig.supxlabel('Gray = outside fixed common support; gold = manual boundary on measured SAG, not predicted segmentation',fontsize=9)
    fig.savefig(output/f'sag_comparison_step_{step:06d}.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,name in zip(axes,names[1:]):
        error=np.where(common,np.abs(images[name]-images['reference']),np.nan)
        im=ax.imshow(error,cmap='magma',vmin=0,vmax=.2)
        ax.set(title=name.upper(),xticks=[],yticks=[])
    fig.colorbar(im,ax=axes,label='Absolute difference (0-1 intensity)',shrink=.8)
    fig.savefig(output/f'sag_error_step_{step:06d}.png',dpi=150);plt.close(fig)


@torch.no_grad()
def render_cor_examples(output,model,data,geometry,step):
    """各里程碑固定展示同四帧，并保存同裁剪的误差和数值。"""
    from .gaussian_reconstruction import BASELINE
    baseline=Baselines(BASELINE)
    plots=output/'plots'/f'step_{step:06d}';plots.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,4,figsize=(12,10),layout='constrained')
    errors,eaxes=plt.subplots(4,3,figsize=(10,10),layout='constrained')
    rows=[];crop=np.s_[120:580,230:750]
    for row,frame in enumerate(data.comparison):
        rendered,*_=model.render(geometry.c2w[frame:frame+1],geometry.config(),1)
        mat=data.initial[frame];xyz=(data.local@mat[:3,:3].T+mat[:3,3]).cpu().numpy()
        values,valid=baseline.sample(xyz)
        valid &= data.mask[frame].bool().cpu().numpy()
        values.update(reference=data.images[frame].cpu().numpy(),gaussian=geometry.to_native(rendered[...,0])[0].cpu().numpy())
        rows.append(dict(frame=frame,**{name:scores(values[name][crop],values['reference'][crop],valid[crop])
            for name in ['knn','inr','gaussian']}))
        for col,name in enumerate(['reference','knn','inr','gaussian']):
            shown=np.where(valid,values[name],np.nan)[crop]
            axes[row,col].imshow(shown,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set(xticks=[],yticks=[])
            if row==0:axes[row,col].set_title(name.upper())
            if col:
                diff=np.where(valid,np.abs(values[name]-values['reference']),np.nan)[crop]
                im=eaxes[row,col-1].imshow(diff,cmap='magma',vmin=0,vmax=.1,interpolation='nearest')
                eaxes[row,col-1].set(xticks=[],yticks=[])
                if row==0:eaxes[row,col-1].set_title(name.upper())
        axes[row,0].set_ylabel(f'COR {frame}');eaxes[row,0].set_ylabel(f'COR {frame}')
    fig.suptitle(f'Patient0 COR | Gaussian {step} | fixed frames, crop and 0-1 range')
    fig.savefig(plots/'cor_comparison.png',dpi=160);plt.close(fig)
    errors.suptitle(f'Patient0 COR absolute difference | Gaussian {step}')
    errors.colorbar(im,ax=eaxes,label='Absolute difference (0-1 intensity)',shrink=.5)
    errors.savefig(plots/'cor_error.png',dpi=150);plt.close(errors)
    write_json(output/'metrics'/f'cor_display_step_{step:06d}.json',dict(crop_yxhw=[120,230,460,520],frames=rows))


@torch.no_grad()
def evaluate(output,model,data,geometry,step,full=False):
    from .gaussian_reconstruction import BASELINE
    model.eval()
    baseline=Baselines(BASELINE)
    assert baseline.grid['poses']==data.metadata['pose_source']
    assert baseline.grid['mask_sha256']==data.metadata['mask_sha256']
    plots=output/'plots'/f'step_{step:06d}';plots.mkdir(parents=True,exist_ok=True)
    ref=data.guidance.reference.cpu().numpy()
    with np.load(BASELINE/'metrics/native_views.npz') as saved:
        common=saved['common'].copy(); truth=saved['truth'].copy()
        images={k:saved[k].copy() for k in ['knn','inr']}
        normalization_error=float(np.max(np.abs(saved['reference']-ref)))
        assert normalization_error<1e-7,normalization_error
        assert np.array_equal(np.rint(saved['reference']*255).astype(np.uint8),np.rint(ref*255).astype(np.uint8))
        # GPU除255可能相差一个float32 ULP；评分统一使用归档参考值。
        ref=saved['reference'].copy()
    torch.cuda.synchronize();tic=time.perf_counter()
    result,*_=model.render(geometry.sag_c2w,geometry.config(sag=True),1)
    gaussian=geometry.to_native(result[...,0])[0].cpu().numpy()
    torch.cuda.synchronize();render_seconds=time.perf_counter()-tic
    assert np.isfinite(gaussian).all()
    images.update(reference=ref,gaussian=gaussian)
    roi=data.guidance.roi if isinstance(data.guidance.roi,dict) else json.loads(Path(data.metadata['roi_config']).read_text())
    native_valid=geometry.to_native(torch.ones_like(result[...,0]))[0].cpu().numpy()>.99999
    assert native_valid[common].all(), ('polar domain loses fixed support',int((common & ~native_valid).sum()))
    out=dict(case='Patient0',step=step,poses_fixed=True,common_sag_pixels=int(common.sum()),reference_normalization_max_error=normalization_error,
             sag={k:scores(v,ref,common) for k,v in images.items() if k!='reference'},roi={},
             gaussian_sag_render_seconds=render_seconds,gaussians=len(model.splats['means']),
             evaluation='native SAG pixels; KNN/INR resliced from archived 0.25 mm grids; Gaussian rendered from a view-dependent field, polar grid then native resampling',
             interpretation='same-case COR fitting and SAG consistency; SAG informed upstream trajectory; different losses and training budgets; no independent 3D truth')
    expected=json.loads((BASELINE/'metrics/comparison.json').read_text())
    for name in ['knn','inr']:
        assert abs(out['sag'][name]['ssim']-expected['sag'][name]['ssim'])<2e-6
    for rid,(x,y,w,h) in zip(roi['roi_ids'],roi['roi_xywh']):
        sl=np.s_[y:y+h,x:x+w]
        out['roi'][rid]={name:scores(value[sl],ref[sl],common[sl]) for name,value in images.items() if name!='reference'}
    edge=truth ^ ndi.binary_erosion(truth)
    band=(ndi.distance_transform_edt(~edge)<=6)&common
    out['cc_boundary_gradient_ncc']={name:_masked_point_metrics(value,ref,band)['gradient_ncc']
                                    for name,value in images.items() if name!='reference'}
    comparison_plot(plots,images,common,truth,roi,step)
    np.savez_compressed(output/'predictions'/f'sag_step_{step:06d}.npz',**images,common=common,truth=truth,
                        poses=data.initial.cpu().numpy())
    # 原生连续KNN/INR另作分辨率对照，避免把体素平滑收益混作模型收益。
    with np.load(BASELINE/'metrics/native_sag.npz') as native:
        out['native_continuous_sag']={name:scores(native[name],ref,common) for name in ['knn','inr']}
        out['native_continuous_sag']['gaussian']=out['sag']['gaussian']
    if full:
        starts=fixed_patches(data)
        rows=[]
        for frame in range(len(data.images)):
            result,*_=model.render(geometry.c2w[frame:frame+1],geometry.config(),1)
            g=geometry.to_native(result[...,0])[0].cpu().numpy()
            source=data.images[frame].cpu().numpy()
            frame_scores={k:[] for k in ['knn','inr','gaussian','polar_roundtrip_train']}
            control=geometry.to_native(geometry.to_polar(data.images[frame:frame+1],2))[0].cpu().numpy()
            for r,c in starts[2*frame:2*frame+2]:
                sl=np.s_[r:r+48,c:c+48]
                local=data.local[sl]
                mat=data.initial[frame]
                xyz=(local@mat[:3,:3].T+mat[:3,3]).cpu().numpy()
                values,valid=baseline.sample(xyz)
                values.update(gaussian=g[sl],polar_roundtrip_train=control[sl])
                for name,v in values.items():frame_scores[name].append(scores(v,source[sl],valid))
            rows.append(dict(frame=frame,**{name:{metric:float(np.mean([x[metric] for x in vals]))
                for metric in ['ssim','psnr','mae','gradient_ncc']} for name,vals in frame_scores.items()}))
            if frame%40==0:print('COR_EVALUATION',frame,'/242',flush=True)
        out['cor']={name:{metric:float(np.mean([r[name][metric] for r in rows]))
                    for metric in ['ssim','psnr','mae','gradient_ncc']} for name in frame_scores}
        out['cor_protocol']=dict(frames=242,patches_per_frame=2,patch_size=48,seed=94117,
            meaning='reconstruction of training source frames, not held-out prediction')
        write_json(output/'metrics'/f'cor_frames_step_{step:06d}.json',rows)
        # 独立读取结束checkpoint，核对权重及位姿，不把可读取当作质量改善。
        ck=torch.load(output/'checkpoints'/f'step_{step:06d}.pt',map_location='cpu',weights_only=False)
        assert ck['completed_steps']==step and torch.equal(ck['poses'],data.initial.cpu())
        for k,v in model.splats.state_dict().items(): assert torch.equal(v.cpu(),ck['splats'][k])
        out['checkpoint_readback']='weights/poses/config/optimizer/scheduler/random states present and weights exact'
        assert all(k in ck for k in ['optimizers','schedulers','random_state','numpy_state','torch_state','cuda_state'])
    render_cor_examples(output,model,data,geometry,step)
    write_json(output/'metrics'/f'comparison_step_{step:06d}.json',out)
    print('COMPARISON',step,json.dumps(out['sag']),flush=True)
    model.train()
    return out
