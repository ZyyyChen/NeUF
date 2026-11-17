"""Paired KNN/INR differences at fixed physical coordinates; no new training."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def summary(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


def ssim(a, b, mask):
    smooth = lambda x: ndi.gaussian_filter(x, 1.5, truncate=5/1.5)
    ma, mb = smooth(a), smooth(b)
    va, vb = np.maximum(smooth(a*a)-ma*ma, 0), np.maximum(smooth(b*b)-mb*mb, 0)
    cv = smooth(a*b)-ma*mb
    values = ((2*ma*mb+.01**2)*(2*cv+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    valid = ndi.minimum_filter(mask, size=11, mode='constant', cval=0)
    return float(values[valid].mean()), int(valid.sum())


def pair(a, b, mask, pitch):
    """a INR, b KNN; differences are symmetric and are not truth errors."""
    m = ndi.binary_erosion(mask, iterations=10)
    d = a[m].astype(np.float64)-b[m]
    score, n = ssim(a, b, mask)
    row = dict(pixels=int(m.sum()), pair_ssim=score, ssim_windows=n,
               mae=float(np.abs(d).mean()), rms=float(np.sqrt(np.mean(d*d))),
               bias_inr_minus_knn=float(d.mean()), p95_abs=float(np.percentile(np.abs(d), 95)),
               intensity_ncc=float(np.corrcoef(a[m], b[m])[0, 1]))
    for name, v in (('inr', a), ('knn', b)):
        gradient = np.hypot(ndi.sobel(v, 0), ndi.sobel(v, 1))/(8*pitch)
        row[name] = dict(mean=float(v[m].mean()), std=float(v[m].std()),
                         gradient_rms=float(np.sqrt(np.mean(gradient[m]**2))), hf_rms={})
        for scale in (.12, .25, .5):
            hf = v-ndi.gaussian_filter(v, scale/pitch)
            row[name]['hf_rms'][str(scale)] = float(np.sqrt(np.mean(hf[m]**2)))
    return row


def source_edge_profiles(volumes, root):
    from neuf.edge_field.trajectory_reconstruction import Inputs, TrajectoryData
    from neuf.edge_field.synthetic_experiment import Volume
    data = TrajectoryData(Inputs('cpu'), 'CNN3D')
    source_grid = json.loads((volumes/'grid.json').read_text())
    assert data.metadata['pose_source'] == source_grid['poses']
    assert data.metadata['mask_sha256'] == source_grid['mask_sha256']
    samplers = {name: Volume(volumes, name) for name in ('inr', 'knn')}
    pitch = data.spacing[0]
    x = np.arange(-9, 9.001, .25)
    tangent = np.array([-2., -1., 0., 1., 2.])
    rows, selected, example_profiles = [], 0, []

    def measure(profile):
        sm = ndi.gaussian_filter1d(profile, .6/.25)
        low, high = sm[x < -6].mean(), sm[x > 6].mean()
        contrast = high-low
        if contrast < .02:
            return None
        q = (sm-low)/contrast
        crossings = []
        for level in (.1, .5, .9):
            ids = np.flatnonzero((q[:-1] <= level) & (q[1:] > level))
            if not len(ids):
                return None
            positions = x[ids]+.25*(level-q[ids])/(q[ids+1]-q[ids])
            crossings.append(float(positions[np.argmin(np.abs(positions))]))
        if not crossings[0] < crossings[1] < crossings[2]:
            return None
        return dict(width_mm=(crossings[2]-crossings[0])*pitch,
                    center_mm=crossings[1]*pitch, contrast=float(contrast),
                    negative_variation=float(-np.minimum(np.diff(q), 0).sum()))

    for frame in range(len(data.images)):
        image = data.images[frame].numpy()
        mask = ndi.binary_erosion(data.mask[frame].numpy().astype(bool), iterations=15)
        sm = ndi.gaussian_filter(image, 1.)
        gy, gx = np.gradient(sm)
        magnitude = np.hypot(gx, gy)
        peaks = mask & (magnitude == ndi.maximum_filter(magnitude, size=17))
        points = np.column_stack(np.nonzero(peaks))
        ranked = np.argsort(magnitude[peaks])[::-1][:40]
        candidates = points[ranked]
        if not len(candidates):
            continue
        xy = candidates[:, ::-1].astype(float)
        gradient = np.column_stack((gx[tuple(candidates.T)], gy[tuple(candidates.T)]))
        normal = gradient/np.linalg.norm(gradient, axis=1)[:, None]
        tan = np.column_stack((-normal[:, 1], normal[:, 0]))
        pixel = xy[:, None, None, :] + normal[:, None, None, :]*x[None, None, :, None] + tan[:, None, None, :]*tangent[None, :, None, None]
        observed = ndi.map_coordinates(image, [pixel[..., 1], pixel[..., 0]], order=1, prefilter=False).mean(1)
        local = np.stack(((pixel[..., 1]-353.5)*pitch, (pixel[..., 0]-472)*pitch, pixel[..., 0]*0), -1)
        matrix = data.initial[frame].numpy()
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        estimates = {name: sampler.sample(world).mean(1) for name, sampler in samplers.items()}
        for j, p in enumerate(observed):
            ref = measure(p)
            # Only measured source selects edges: high contrast, one dominant monotonic transition.
            if ref is None or ref['contrast'] < .06 or ref['negative_variation'] > .25 or abs(ref['center_mm']) > 3*pitch:
                continue
            selected += 1
            records = {name: measure(v[j]) for name, v in estimates.items()}
            row = dict(frame=frame, row=int(candidates[j, 0]), column=int(candidates[j, 1]), measured=ref)
            for name, result in records.items():
                if result is not None:
                    result['width_ratio_to_measured'] = result['width_mm']/ref['width_mm']
                    result['center_error_mm'] = result['center_mm']-ref['center_mm']
                    result['contrast_ratio_to_measured'] = result['contrast']/ref['contrast']
                row[name] = result
            rows.append(row)
            if records['inr'] is not None and records['knn'] is not None:
                example_profiles.append(dict(frame=frame, row=int(candidates[j, 0]), column=int(candidates[j, 1]),
                                             measured=p.tolist(), inr=estimates['inr'][j].tolist(), knn=estimates['knn'][j].tolist()))
        if frame % 50 == 0:
            print('edge profiles', frame, 'selected', selected, flush=True)
    common = [r for r in rows if r['inr'] is not None and r['knn'] is not None]
    result = dict(selection='top40 independent measured COR gradient peaks/frame, NMS17; support eroded15px; contrast>=.06, negative variation<=.25, center within3px; profile normal +/-9px, tangent average5px, Gaussian sigma.6px',
                  selected=selected, common_valid=len(common), sampled_frames=242,
                  method_valid={name: sum(r[name] is not None for r in rows) for name in ('inr', 'knn')})
    for name in ('inr', 'knn'):
        result[name] = {key: summary([r[name][key] for r in common]) for key in ('width_mm', 'width_ratio_to_measured', 'center_error_mm', 'contrast_ratio_to_measured')}
        result[name]['absolute_center_error_mm'] = summary([abs(r[name]['center_error_mm']) for r in common])
    result['measured_width_mm'] = summary([r['measured']['width_mm'] for r in common])
    result['inr_minus_knn_width_mm'] = summary([r['inr']['width_mm']-r['knn']['width_mm'] for r in common])
    result['inr_wider_fraction'] = float(np.mean([r['inr']['width_mm'] > r['knn']['width_mm'] for r in common]))
    write(root/'edge_profiles.json', dict(summary=result, rows=rows, examples=example_profiles, normal_pixels=x.tolist()))
    return result


def run(args):
    from neuf.edge_field.synthetic_experiment import Volume, digest
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    assert digest(args.dataset/'acquisition.h5') == manifest['acquisition_sha256']
    samplers = {name: Volume(args.volumes, name) for name in ('inr', 'knn', 'support')}
    for name in samplers:
        assert digest(args.volumes/f'{name}.raw') == manifest['source_raw_sha256'][name]
    frame_rows, snapshots = [], {}
    with h5py.File(args.dataset/'acquisition.h5', 'r') as h:
        for frame in range(len(h['mask'])):
            a, b, mask = h['inr/images'][frame], h['knn/images'][frame], h['mask'][frame]
            row = pair(a, b, mask, manifest['source_spacing_hw_mm'][0])
            row['frame'] = frame
            frame_rows.append(row)
            if frame in (26, 126, 226):
                snapshots.update({f'f{frame}_inr': a, f'f{frame}_knn': b, f'f{frame}_mask': mask})
            if frame % 40 == 0:
                print('paired frame', frame, row['pair_ssim'], flush=True)
        for name in ('inr', 'knn'):
            snapshots[f'sag_{name}'] = h[f'{name}/reference'][:]
        snapshots['sag_mask'] = h['reference_mask'][:]
    np.savez_compressed(root/'snapshots.npz', **snapshots)
    write(root/'new_trajectory_frames.json', frame_rows)
    result = dict(geometry='same true matrices and native grid for both teachers; noise does not enter rendered images',
                  source_volume_sha256=manifest['source_raw_sha256'],
                  sampled_frames={k: summary([r[k] for r in frame_rows]) for k in ('pair_ssim', 'mae', 'rms', 'bias_inr_minus_knn', 'p95_abs', 'intensity_ncc')},
                  frame_ratios={})
    for metric in ('gradient_rms', '.12', '.25', '.5'):
        values = [(r['inr'][metric]/r['knn'][metric] if metric == 'gradient_rms' else r['inr']['hf_rms'][str(float(metric))]/r['knn']['hf_rms'][str(float(metric))]) for r in frame_rows]
        result['frame_ratios'][metric] = dict(summary(values), inr_lower_frames=int(np.sum(np.array(values)<1)))
    # Full-volume comparisons exclude the arbitrary support exterior; mm-based differences use identical voxels.
    mask = ndi.minimum_filter(samplers['support'].values, size=17, mode='constant', cval=0).astype(bool)
    a, b = samplers['inr'].values, samplers['knn'].values
    d = a[mask].astype(np.float64)-b[mask]
    result['volume'] = dict(interior_voxels=int(mask.sum()), support_voxels=int(samplers['support'].values.sum()),
                            border_exclusion='complete 17x17x17 support; 2mm margin along each voxel axis', bias=float(d.mean()), mae=float(np.abs(d).mean()),
                            rms=float(np.sqrt(np.mean(d*d))), p50_abs=float(np.median(np.abs(d))),
                            p95_abs=float(np.percentile(np.abs(d), 95)), p99_abs=float(np.percentile(np.abs(d), 99)),
                            fractions_abs_larger={str(t):float(np.mean(np.abs(d)>t)) for t in (.01, .02, .05, .1)},
                            intensity_ncc=float(np.corrcoef(a[mask], b[mask])[0, 1]), methods={}, lowpass_difference={})
    del d
    for name in ('inr', 'knn'):
        v = samplers[name].values
        stat = dict(mean=float(v[mask].mean()), std=float(v[mask].std()), hf_rms={}, gradient_rms_per_mm={})
        for scale in (.25, .5):
            sm = ndi.gaussian_filter(v, scale/samplers[name].spacing)
            residual = v[mask]-sm[mask]
            stat['hf_rms'][str(scale)] = float(np.sqrt(np.mean(residual**2)))
            if name == 'inr':
                np.save(root/f'lowpass_inr_{scale}.npy', sm[mask])
            else:
                difference = np.load(root/f'lowpass_inr_{scale}.npy')-sm[mask]
                result['volume']['lowpass_difference'][str(scale)] = float(np.mean(difference**2)/result['volume']['rms']**2)
                (root/f'lowpass_inr_{scale}.npy').unlink()
            del sm, residual
        for axis, direction in enumerate('zyx'):
            lo, hi, mid = [slice(None)]*3, [slice(None)]*3, [slice(None)]*3
            lo[axis], hi[axis], mid[axis] = slice(None, -2), slice(2, None), slice(1, -1)
            grad = (v[tuple(hi)]-v[tuple(lo)])/(2*samplers[name].spacing)
            stat['gradient_rms_per_mm'][direction] = float(np.sqrt(np.mean(grad[mask[tuple(mid)]]**2)))
            del grad
        result['volume']['methods'][name] = stat
        print('volume', name, stat, flush=True)
    write(root/'comparison.json', result)
    result['source_edge_profiles'] = source_edge_profiles(args.volumes, root)
    old = json.loads((args.volumes.parent/'metrics/comparison.json').read_text())
    result['original_measured_reference'] = {key: old[key] for key in ('cor', 'sag', 'native_continuous_sag', 'roi', 'cc_boundary_band', 'cc_2d_reference', 'interpretation')}
    result['limits'] = ['single patient, one INR seed, fixed source poses', 'new trajectory compares two teachers; neither is used as independent anatomical truth', 'high frequency can be texture, noise or interpolation artifacts', 'original SAG informed upstream pose fitting', 'edge profiles use measured COR and finite 0.25mm exported volumes, not clinical resolution calibration']
    write(root/'comparison.json', result)
    print('analysis complete', flush=True)


def sag_shared_error(volumes):
    with np.load(volumes.parent/'metrics/native_views.npz') as h:
        m=h['common'].astype(bool)
        ei=h['inr'][m].astype(np.float64)-h['reference'][m]
        ek=h['knn'][m].astype(np.float64)-h['reference'][m]
    shared=(ei+ek)/2
    difference=(ei-ek)/2
    total=(np.mean(ei**2)+np.mean(ek**2))/2
    return dict(common_pixels=int(m.sum()), error_correlation=float(np.corrcoef(ei,ek)[0,1]),
        inr_rmse=float(np.sqrt(np.mean(ei**2))),knn_rmse=float(np.sqrt(np.mean(ek**2))),
        pair_rmse=float(np.sqrt(np.mean((ei-ek)**2))),
        shared_component_mse_fraction=float(np.mean(shared**2)/total),
        method_difference_component_mse_fraction=float(np.mean(difference**2)/total),
        definition='eI=INR-measured, eK=KNN-measured; shared=(eI+eK)/2, method_difference=(eI-eK)/2; mean of two MSE equals the two component energies exactly; descriptive overlap, not a causal identification')


def secondary(args):
    import hashlib
    import torch
    from neuf.edge_field.synthetic_experiment import Volume
    from neuf.edge_field.trajectory_reconstruction import Inputs, TrajectoryData
    from neuf.edge_field.coronal_features import coronal_projection
    from neuf.edge_field.losses import masked_ssim
    sys.path.insert(0, str(args.volumes.parents[1]/'20261110_neuf_residual_blur/run_config'))
    from frame_patches import get_patches
    torch.set_num_threads(4)
    data = TrajectoryData(Inputs('cpu'), 'CNN3D')
    frames, local, reference, starts = get_patches(data)
    old = json.loads((args.volumes.parent/'metrics/comparison.json').read_text())
    assert hashlib.sha256(np.ascontiguousarray(starts, dtype=np.int64).tobytes()).hexdigest() == old['patches']['starts_hash']
    matrices = data.initial[frames]
    xyz = (torch.einsum('nij,nhwj->nhwi', matrices[:, :3, :3], local)+matrices[:, None, None, :3, 3]).numpy()
    images = {'measured': reference.numpy(), **{name:Volume(args.volumes, name).sample(xyz) for name in ('inr', 'knn')}}
    results = dict(source_patches=old['patches'], frequency={}, source_distance_strata=[])
    for sigma in (1., 2.):
        high = {name:(v-ndi.gaussian_filter(v, (0, sigma, sigma)))[:, 10:-10, 10:-10] for name, v in images.items()}
        y = high['measured'].astype(np.float64).ravel(); y -= y.mean()
        stats = {}
        for name in ('inr', 'knn'):
            x = high[name].astype(np.float64).ravel(); x -= x.mean()
            slope = float(np.dot(x, y)/np.dot(y, y))
            stats[name] = dict(rms_ratio=float(np.sqrt(np.dot(x,x)/np.dot(y,y))),
                               ncc=float(np.corrcoef(x,y)[0,1]), aligned_amplitude=slope,
                               unaligned_rms_ratio=float(np.sqrt(np.mean((x-slope*y)**2)/np.mean(y*y))),
                               relative_hf_mse=float(np.mean((x-y)**2)/np.mean(y*y)))
            per_frame=[]
            for a,b in zip(high[name].reshape(242,-1), high['measured'].reshape(242,-1)):
                per_frame.append(float(np.corrcoef(a,b)[0,1]))
            stats[name]['per_frame_ncc']=per_frame
            stats[name]['frame_mean_ncc']=float(np.mean(per_frame))
        results['frequency'][str(sigma*data.spacing[0])] = stats
    grad=np.hypot(ndi.sobel(images['measured'],axis=1),ndi.sobel(images['measured'],axis=2))[:,10:-10,10:-10]/8
    energy=np.sqrt((grad**2).reshape(242,-1).mean(1))
    quantiles=np.quantile(energy,[0,.25,.5,.75,1.])
    old_frames=json.loads((args.volumes.parent/'metrics/cor_frames.json').read_text())
    results['source_texture_quartiles']=[]
    for q in range(4):
        take=(energy>=quantiles[q])&(energy<quantiles[q+1] if q<3 else energy<=quantiles[q+1])
        results['source_texture_quartiles'].append(dict(quartile=q+1,frames=int(take.sum()),
            measured_gradient_rms_bounds=quantiles[q:q+2].tolist(),
            **{name:{key:float(np.mean([r[key] for r,keep in zip(old_frames[name],take) if keep])) for key in ('gradient_ncc','ssim','mse')} for name in ('inr','knn')}))
    with np.load(args.dataset/'truth.npz') as h:
        points = torch.from_numpy(h['holdout_xyz'])
        delta = h['inr'].astype(np.float64)-h['knn']
    distances=[]
    with torch.no_grad():
        for points_chunk in points.split(2048):
            projected, weights, valid = coronal_projection(points_chunk, data.initial, data.local, data.mask)
            distance=(projected-points_chunk[None]).norm(dim=-1).masked_fill(weights<=0,float('inf')).amin(0)
            distances.append(distance.numpy())
    distances=np.concatenate(distances)
    bins=[0., .03, .06, .125, .25, .5, 1., 2.25]
    for lo,hi in zip(bins[:-1],bins[1:]):
        selected=(distances>=lo)&(distances<hi)
        if selected.any():
            results['source_distance_strata'].append(dict(lo_mm=lo,hi_mm=hi,count=int(selected.sum()),
                pair_mae=float(np.abs(delta[selected]).mean()), pair_rms=float(np.sqrt(np.mean(delta[selected]**2)))))
    with h5py.File(args.dataset/'acquisition.h5','r') as h:
        a,b,m=(h[key][126] for key in ('inr/images','knn/images','mask'))
    value,count=masked_ssim(torch.from_numpy(a)[None,None],torch.from_numpy(b)[None,None],torch.from_numpy(m)[None,None])
    independent,n=ssim(a,b,m)
    assert n==count and abs(float(value)-independent)<5e-5
    results['ssim_validation']=dict(scipy=independent,torch=float(value),same_complete_windows=n,absolute_error=abs(float(value)-independent))
    with np.load(args.volumes.parent/'metrics/native_views.npz') as h:
        a,b=h['masks_inr'].astype(bool),h['masks_knn'].astype(bool)
    results['sag_shared_error']=sag_shared_error(args.volumes)
    results['segmentation_pair']=dict(dice=float(2*(a&b).sum()/(a.sum()+b.sum())),
        disagreeing_pixels=int((a^b).sum()), union_pixels=int((a|b).sum()))
    rows=json.loads((args.output/'new_trajectory_frames.json').read_text())
    results['acquisition_regions']={name:{key:summary([r[key] for r in rows[selection]]) for key in ('pair_ssim','mae','rms')}
        for name,selection in (('first24',slice(0,24)),('middle194',slice(24,218)),('last24',slice(218,242)))}
    results['worst_frames_by_mae']=[{'frame':r['frame'],'mae':r['mae'],'pair_ssim':r['pair_ssim']} for r in sorted(rows,key=lambda r:r['mae'],reverse=True)[:10]]
    write(args.output/'detail_attribution.json',results)
    print(json.dumps(results,indent=2),flush=True)


def figures(args):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':9,
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
    root = args.output
    with np.load(root/'snapshots.npz') as h:
        images = {k:h[k] for k in h.files}
    gray = plt.get_cmap('gray').copy(); gray.set_bad('#dddddd')
    diverging = plt.get_cmap('RdBu_r').copy(); diverging.set_bad('#dddddd')

    def save(fig, name, exclude=()):
        require_matplotlib_panel_alignment(fig, json_out=root/f'{name}.alignment.json', strict=True, exclude_axes=exclude)
        fig.savefig(root/f'{name}.png', dpi=300)
        fig.savefig(root/f'{name}.pdf')
        fig.savefig(root/f'{name}.svg')
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(10, 8.6), layout='constrained')
    crop = np.s_[200:440, 320:640]
    for row, frame in enumerate((26, 126, 226)):
        a, b, mask = [images[f'f{frame}_{key}'][crop] for key in ('inr', 'knn', 'mask')]
        for col, panel in enumerate((a, b, a-b)):
            im = axes[row, col].imshow(np.where(mask, panel, np.nan), cmap=gray if col<2 else diverging,
                                      vmin=0 if col<2 else -.04, vmax=.6 if col<2 else .04, interpolation='nearest')
            axes[row, col].set(xticks=[], yticks=[])
        axes[row, 0].set_ylabel(f'New COR #{frame}\n38.4 x 28.8 mm')
    for ax, title in zip(axes[0], ('INR', 'KNN', 'INR minus KNN')):
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes[:, 2], shrink=.72, pad=.02, label='Intensity difference (range 0-1)')
    fig.suptitle('Identical new trajectory: fixed center crops; shared gray window 0-0.6', fontsize=11)
    save(fig, 'new_trajectory_details', [cb.ax])

    from matplotlib.patches import Rectangle
    frame_rows=json.loads((root/'new_trajectory_frames.json').read_text())
    worst=max(frame_rows,key=lambda r:r['mae'])['frame']
    with h5py.File(args.dataset/'acquisition.h5','r') as h:
        a,b,mask=(h[key][worst] for key in ('inr/images','knn/images','mask'))
    safe=ndi.minimum_filter(mask,size=(96,128),mode='constant',cval=0)
    ranking=ndi.uniform_filter(np.abs(a-b),size=(96,128)).astype(np.float64)
    ranking[~safe]=-np.inf
    y,x=np.unravel_index(np.argmax(ranking),ranking.shape)
    y,x=int(y)-48,int(x)-64
    crop=np.s_[y:y+96,x:x+128]
    write(root/'largest_difference_selection.json',dict(frame=worst,box_xywh=[x,y,128,96],
        rule='frame with largest paired MAE among all242; complete-support 128x96 crop with largest mean absolute difference; intentionally a worst-case view'))
    fig,axes=plt.subplots(2,3,figsize=(10,6.5),layout='constrained')
    for row,selection in enumerate((np.s_[:,:],crop)):
        for col,panel in enumerate((a,b,a-b)):
            im=axes[row,col].imshow(np.where(mask,panel,np.nan)[selection],cmap=gray if col<2 else diverging,
                vmin=0 if col<2 else -.04,vmax=.6 if col<2 else .04,interpolation='nearest')
            axes[row,col].set(xticks=[],yticks=[])
            if row==0:
                axes[row,col].add_patch(Rectangle((x,y),128,96,fill=False,edgecolor='#c28e36',linewidth=1.1))
        axes[row,0].set_ylabel(f'New COR #{worst}' if row==0 else 'Largest-difference crop\n15.4 x 11.5 mm')
    for ax,title in zip(axes[0],('INR','KNN','INR minus KNN')):
        ax.set_title(title)
    cb=fig.colorbar(im,ax=axes[:,2],shrink=.7,pad=.02,label='Intensity difference (range 0-1)')
    fig.suptitle('Boundary case selected by largest paired difference; shared display window 0-0.6',fontsize=11)
    save(fig,'largest_difference',[cb.ax])

    with np.load(args.volumes.parent/'metrics/native_views.npz') as h:
        views = {k:h[k] for k in h.files}
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.3), layout='constrained')
    for row, (label, box) in enumerate((('R1', (320,207,314,235)), ('R5', (458,387,227,179)))):
        x, y, width, height = box
        for col, name in enumerate(('reference', 'inr', 'knn', 'difference')):
            base = views['inr']-views['knn'] if name == 'difference' else views[name]
            panel = np.full((235, 314), np.nan, dtype=np.float32)
            valid = views['common'][y:y+height, x:x+width]
            panel[:height, :width] = np.where(valid, base[y:y+height,x:x+width], np.nan)
            im = axes[row,col].imshow(panel, cmap=gray if col<3 else diverging,
                                     vmin=0 if col<3 else -.04, vmax=.6 if col<3 else .04, interpolation='nearest')
            axes[row,col].set(xticks=[], yticks=[])
        axes[row,0].set_ylabel(f'Measured SAG {label}')
    for ax, title in zip(axes[0], ('Measured', 'INR', 'KNN', 'INR minus KNN')):
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes[:, 3], shrink=.72, pad=.02, label='Intensity difference')
    fig.suptitle('Fixed manual SAG ROIs; shared gray window 0-0.6; gray padding is excluded', fontsize=11)
    save(fig, 'measured_sag_details', [cb.ax])

    rows = json.loads((root/'new_trajectory_frames.json').read_text())
    edges = json.loads((root/'edge_profiles.json').read_text())
    valid = [r for r in edges['rows'] if r['inr'] is not None and r['knn'] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    axes[0].plot([r['frame'] for r in rows], [r['pair_ssim'] for r in rows], color='#526d86')
    axes[0].set(xlabel='New acquisition frame', ylabel='INR/KNN pair SSIM', title='Structural agreement')
    axes[1].plot([r['frame'] for r in rows], [r['inr']['hf_rms']['0.25']/r['knn']['hf_rms']['0.25'] for r in rows], color='#526d86')
    axes[1].axhline(1., color='#a87144', linestyle='--', linewidth=1)
    axes[1].set(xlabel='New acquisition frame', ylabel='INR / KNN high-frequency RMS', title='Texture at Gaussian sigma 0.25 mm')
    for name, color in (('inr', '#526d86'), ('knn', '#a87144')):
        values = np.sort([r[name]['width_ratio_to_measured'] for r in valid])
        axes[2].plot(values, np.arange(1,len(values)+1)/len(values), color=color, label=name.upper())
    axes[2].set(xlabel='Edge width / measured COR width', ylabel='Fraction of matched profiles', title=f'10-90% width: {len(valid)} matched edges')
    axes[2].legend(loc='lower right')
    save(fig, 'paired_quantification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'secondary', 'figures'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--volumes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args)
