"""User-selected CC-head checks: images, atlas limitations, and rendering controls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .paired_volume_analysis import ssim, write
from .synthetic_experiment import Volume


HEAD = (330, 215, 135, 125)
TIGHT = (340, 220, 85, 85)
PITCH = .133849221834


def region(box, shape):
    x,y,w,h=box
    mask=np.zeros(shape,dtype=bool); mask[y:y+h,x:x+w]=True
    return mask


def crop(a, box):
    x,y,w,h=box
    return a[y:y+h,x:x+w]


def image_scores(a,b,m):
    score,count=ssim(a,b,m)
    delta=a[m].astype(np.float64)-b[m]
    return dict(ssim=score,windows=count,pixels=int(m.sum()),mse=float(np.mean(delta**2)),
                mae=float(np.abs(delta).mean()),ncc=float(np.corrcoef(a[m],b[m])[0,1]))


def main(args):
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    with np.load(args.source/'metrics/native_views.npz') as f:
        views={k:f[k] for k in f.files}
    with np.load(args.source/'metrics/native_sag.npz') as f:
        direct={k:f[k] for k in f.files}
    # Fixed original ROI identity remains mandatory; the added crops follow the user's head location.
    from PIL import Image
    workspace=args.source.parents[1]
    roi=json.loads((workspace/'EdgeReg/configs/patient0_rois.json').read_text())
    reference_path=workspace/roi['image_path']
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==roi['image_sha256']
    assert np.array_equal(np.asarray(Image.open(reference_path).convert('L')),np.rint(255*views['reference']).astype(np.uint8))
    ref=views['reference']; truth=views['truth'].astype(bool); common=views['common'].astype(bool)
    result=dict(head_box_xywh=HEAD,tight_box_xywh=TIGHT,pitch_mm=PITCH,
        scope='user-requested CC-head diagnostic inside existing R1; does not replace the fixed R1-R5 protocol',
        manual_contour='manually annotated measured SAG, transferred at identical coordinates; no new manual reconstruction labels',regions={})
    truth_edge=truth ^ ndi.binary_erosion(truth)
    for title,box in (('head',HEAD),('head_tight',TIGHT),('R1',(320,207,314,235))):
        selection=region(box,ref.shape); valid=common&selection
        statistics={}
        for name in ('inr','knn'):
            pred=views['masks_'+name].astype(bool)
            pred_edge=pred ^ ndi.binary_erosion(pred)
            td=ndi.distance_transform_edt(~pred_edge)[truth_edge&selection]*PITCH
            pd=ndi.distance_transform_edt(~truth_edge)[pred_edge&selection]*PITCH
            gt_area=(truth&selection).sum(); pr_area=(pred&selection).sum()
            gt_x=np.nonzero(truth&selection)[1]; pred_x=np.nonzero(pred&selection)[1]
            direct_valid=valid & direct['support'].astype(bool)
            statistics[name]=dict(image=image_scores(crop(views[name],box),crop(ref,box),crop(valid,box)),
                atlas_segmentation=dict(dice=float(2*(truth&pred&selection).sum()/(gt_area+pr_area)),
                    fn=int((truth&~pred&selection).sum()),fp=int((~truth&pred&selection).sum()),
                    truth_pixels=int(gt_area),predicted_pixels=int(pr_area),
                    boundary_assd_mm=float((td.mean()+pd.mean())/2),boundary_hd95_mm=float(max(np.percentile(td,95),np.percentile(pd,95))),
                    leftmost_offset_mm=float((pred_x.min()-gt_x.min())*PITCH),
                    warning='atlas-based segmentation has anatomical prior; these contours are not independent evidence of image-shape fidelity'),
                direct_continuous=image_scores(crop(direct[name],box),crop(ref,box),crop(direct_valid,box)),
                volume_vs_direct=image_scores(crop(views[name],box),crop(direct[name],box),crop(direct_valid,box)))
        statistics['inr_vs_knn']=image_scores(crop(views['inr'],box),crop(views['knn'],box),crop(valid,box))
        result['regions'][title]=statistics
    # Search is diagnostic fitting to this same ROI, not an independently validated pose correction.
    box=TIGHT;x,y,w,h=box
    safe=ndi.minimum_filter(common,size=25,mode='constant',cval=0).astype(bool)
    m=crop(safe,box); target=crop(ref,box)
    result['translation_probe']={'search_pixels':[-12,12],'fixed_pixels':int(m.sum()),
        'interpretation':'best local in-plane translation fitted to this same measured head; no model or trajectory is changed'}
    for name in ('inr','knn'):
        baseline=image_scores(crop(views[name],box),target,m)
        best=dict(dx=0,dy=0,**baseline)
        for dy in range(-12,13):
            for dx in range(-12,13):
                candidate=views[name][y+dy:y+h+dy,x+dx:x+w+dx]
                score,count=ssim(candidate,target,m)
                if score>best['ssim']:
                    best=dict(dx=dx,dy=dy,**image_scores(candidate,target,m))
        result['translation_probe'][name]=dict(baseline=baseline,best=best,
            sampling_offset_mm=[best['dx']*PITCH,best['dy']*PITCH])
    # Test whether an adjacent parallel plane contains the missing local appearance.
    volumes={name:Volume(args.source/'volumes',name) for name in ('inr','knn','support')}
    support_float=volumes['support'].values.astype(np.float32)
    yy,xx=np.mgrid[y:y+h,x:x+w]
    xyz=np.stack((xx*PITCH,yy*PITCH,xx*0),axis=-1).astype(np.float32)
    offsets=np.arange(-2,2.001,.25)
    z_images={name:[] for name in ('inr','knn')}; masks=[]
    for z in offsets:
        xyz[...,2]=z
        masks.append(volumes['support'].sample(xyz,support_float)>.999)
        for name in z_images:
            z_images[name].append(volumes[name].sample(xyz))
    zmask=np.logical_and.reduce(masks)&crop(common,box)
    result['z_probe']=dict(offsets_mm=offsets.tolist(),fixed_pixels=int(zmask.sum()),
        interpretation='parallel z-plane search within +/-2mm on the same local target; in-sample diagnostic, no corrected trajectory claim')
    for name in ('inr','knn'):
        rows=[dict(z_mm=float(z),**image_scores(im,target,zmask)) for z,im in zip(offsets,z_images[name])]
        result['z_probe'][name]=dict(rows=rows,best=max(rows,key=lambda r:r['ssim']))
    write(root/'metrics.json',result)
    np.savez_compressed(root/'head_arrays.npz',**{k:crop(v,HEAD) for k,v in views.items()},
        direct_inr=crop(direct['inr'],HEAD),direct_knn=crop(direct['knn'],HEAD),
        z_inr=np.stack(z_images['inr']),z_knn=np.stack(z_images['knn']),z_offsets=offsets)
    print(json.dumps(result,indent=2),flush=True)
    figures(root,views,result)


def figures(root, views, result):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    x,y,w,h=HEAD
    extent=((x-.5)*PITCH,(x+w-.5)*PITCH,(y+h-.5)*PITCH,(y-.5)*PITCH)
    xx=np.arange(x,x+w)*PITCH; yy=np.arange(y,y+h)*PITCH
    fig,axes=plt.subplots(2,3,figsize=(10.5,7.2),layout='constrained',sharex=True,sharey=True)
    for col,(name,title) in enumerate((('reference','Measured SAG'),('inr','INR, fixed 800k'),('knn','KNN, same trajectory'))):
        image=crop(views[name],HEAD)
        for row in range(2):
            axes[row,col].imshow(image,cmap='gray',vmin=0,vmax=.6,interpolation='nearest',extent=extent)
            axes[row,col].set_xlabel('World x [mm]')
        axes[1,col].contour(xx,yy,crop(views['truth'],HEAD),levels=[.5],colors=['#e5ad35'],linewidths=1.)
        axes[0,col].set_title(title)
    axes[0,0].set_ylabel('Raw head crop\nWorld y [mm]')
    axes[1,0].set_ylabel('Same measured contour\nWorld y [mm]')
    fig.suptitle('CC head: identical coordinates, shared gray window 0-0.6\nYellow is the manual measured-SAG contour, not a reconstruction segmentation',fontsize=11)
    save(fig,root/'cc_head_comparison')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for name,color in (('inr','#526d86'),('knn','#a87144')):
        rows=result['z_probe'][name]['rows']
        axes[0].plot([r['z_mm'] for r in rows],[r['ssim'] for r in rows],color=color,label=name.upper())
        base=result['translation_probe'][name]['baseline']['ssim']; best=result['translation_probe'][name]['best']['ssim']
        axes[1].plot([0,1],[base,best],'o-',color=color,label=name.upper())
    axes[0].set(xlabel='Parallel-plane offset z [mm]',ylabel='Head SSIM to measured SAG',title='Adjacent-plane diagnostic')
    axes[1].set(xticks=[0,1],xticklabels=['Original','Best translation'],ylabel='Head SSIM to measured SAG',title='Local translation diagnostic')
    axes[0].legend();axes[1].legend()
    save(fig,root/'head_controls')


def save(fig,path):
    import matplotlib.pyplot as plt
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=path.with_suffix('.alignment.json'),strict=True)
    fig.savefig(path.with_suffix('.png'),dpi=300)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def upstream(args):
    """Locate whether the same local discrepancy predates the INR fit."""
    import h5py
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    root=args.output
    directory=args.source.parents[1]/'logs/20261026_train23/cerebral/index_patient0/CNN3D6D'
    manifest=json.loads((directory/'run_config/input_manifest.json').read_text())
    assert abs(manifest['reference_spacing_mm']-PITCH)<1e-12
    with np.load(args.source/'metrics/native_views.npz') as f:
        views={k:f[k] for k in f.files}
    sagittal=Path(manifest['sagittal']['path'])
    if not sagittal.is_absolute():
        sagittal=args.source.parents[1]/sagittal
    assert hashlib.sha256(sagittal.read_bytes()).hexdigest()==manifest['sagittal']['sha256']
    with h5py.File(sagittal,'r') as h:
        np.testing.assert_allclose(np.asarray(h['data_sag']).T/255,views['reference'],atol=1e-7,rtol=0)
    images={};masks=[]
    for name,step in (('edge_initial',0),('edge_cnn3d',200)):
        with h5py.File(directory/f'predictions/step_{step:06d}.h5','r') as h:
            images[name]=h['image'][:]
            masks.append(h['support'][:].astype(bool))
            assert images[name].shape==views['reference'].shape
            assert 0<=images[name].min()<=images[name].max()<=1
            if step==0:
                assert np.count_nonzero(h['correction'][:])==0
    common=np.logical_and.reduce(masks)&views['common'].astype(bool)
    m=crop(common,TIGHT); ref=crop(views['reference'],TIGHT)
    stats={name:image_scores(crop(im,TIGHT),ref,m) for name,im in {**images,'inr':views['inr'],'knn':views['knn']}.items()}
    upstream_error=crop(images['edge_cnn3d'],TIGHT)[m]-ref[m]
    for name in ('inr','knn'):
        err=crop(views[name],TIGHT)[m]-ref[m]
        stats[name]['error_correlation_with_saved_cnn3d']=float(np.corrcoef(err,upstream_error)[0,1])
    path=root/'metrics.json'; result=json.loads(path.read_text())
    result['upstream']=dict(stages=stats,source=str(directory),
        interpretation='step0 and step200 share the upstream renderer; INR/KNN use their own renderers. This locates the presence of the error before INR, not a clean cross-renderer causal ablation.',
        source_manifest_reference_verified=True)
    write(path,result)
    np.savez_compressed(root/'upstream_arrays.npz',reference=crop(views['reference'],HEAD),
                        **{name:crop(im,HEAD) for name,im in images.items()},inr=crop(views['inr'],HEAD),common=crop(common,HEAD))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(12.5,3.6),layout='constrained')
    panels=[('Measured SAG',views['reference']),('Edge initialization',images['edge_initial']),
            ('After 3D CNN',images['edge_cnn3d']),('INR, fixed 800k',views['inr'])]
    for ax,(title,im) in zip(axes,panels):
        ax.imshow(crop(im,HEAD),cmap='gray',vmin=0,vmax=.6,interpolation='nearest')
        ax.set(title=title,xticks=[],yticks=[])
    fig.suptitle('CC head across saved stages: identical native SAG coordinates',fontsize=11)
    save(fig,root/'upstream_head')
    print(json.dumps(result['upstream'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream-only',action='store_true')
    args=p.parse_args()
    upstream(args) if args.upstream_only else main(args)
