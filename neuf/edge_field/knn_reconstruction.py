"""Inverse-distance compounding on rigidly posed ultrasound frames.

Uses the project's nb_pts=3, multi_img=0 stencil: a 3x3 block on the
nearest frame and on its closer temporal neighbour. The projection and
distances here use each frame's full pose in millimetres.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _nearest_frame(xyz, transforms, masks, result, n, F: tl.constexpr,
                   H: tl.constexpr, W: tl.constexpr, inv_row, inv_col, row0, col0,
                   radius, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(xyz + 3*i, i<n, 0.)
    y = tl.load(xyz + 3*i+1, i<n, 0.)
    z = tl.load(xyz + 3*i+2, i<n, 0.)
    best = tl.full((BLOCK,), float('inf'), tl.float32)
    which = tl.full((BLOCK,), -1, tl.int32)
    for f in range(F):
        p = transforms + 12*f
        depth = x*tl.load(p+8) + y*tl.load(p+9) + z*tl.load(p+10) + tl.load(p+11)
        close = (i<n) & (tl.abs(depth)<=radius) & (depth*depth<=best)
        if tl.sum(close.to(tl.int32),0)>0:
            row = (x*tl.load(p) + y*tl.load(p+1) + z*tl.load(p+2) + tl.load(p+3) - row0)*inv_row
            col = (x*tl.load(p+4) + y*tl.load(p+5) + z*tl.load(p+6) + tl.load(p+7) - col0)*inv_col
            r = tl.floor(row + .5).to(tl.int32)
            c = tl.floor(col + .5).to(tl.int32)
            inside = close & (r>=0) & (r<H) & (c>=0) & (c<W)
            valid = tl.load(masks + (f*H+r)*W+c, inside, 0) != 0
            dy = (row-r)/inv_row
            dx = (col-c)/inv_col
            dist = dx*dx + dy*dy + depth*depth
            take = valid & (dist<best)
            best = tl.where(take,dist,best)
            which = tl.where(take,f,which)
    tl.store(result+i,which,i<n)


class RigidKNN:
    def __init__(self, data, radius_mm=2.25):
        self.images = data.images
        self.mask = data.mask.to(torch.uint8).contiguous()
        self.matrices = data.initial
        rotation = self.matrices[:,:3,:3].transpose(1,2)
        offset = -(rotation @ self.matrices[:,:3,3:])
        self.inverse = torch.cat((rotation,offset),dim=-1).contiguous()
        self.spacing = tuple(map(float,data.spacing))
        self.origin = tuple(map(float,data.local[0,0,:2]))
        self.radius_mm = radius_mm
        self.neighbour_offsets = torch.tensor([-1,0,1],device=self.images.device)
        self.sr, self.sc = self.spacing
        self.r0, self.c0 = self.origin

    def project(self, xyz, frames):
        T = self.inverse[frames]
        local = (T[:,:,:3]*xyz[:,None,:]).sum(-1) + T[:,:,3]
        row = (local[:,0]-self.r0)/self.sr
        col = (local[:,1]-self.c0)/self.sc
        return row,col,local[:,2]

    def center(self, xyz, frames):
        total,h,w = self.images.shape
        ids = frames.clamp(0,total-1)
        row,col,normal = self.project(xyz,ids)
        r = torch.floor(row+.5).long()
        c = torch.floor(col+.5).long()
        good = ((frames==ids) & (normal.abs()<=self.radius_mm)
                & (r>=0) & (r<h) & (c>=0) & (c<w)
                & self.mask[ids,r.clamp(0,h-1),c.clamp(0,w-1)].bool())
        d2 = ((row-r)*self.sr).square()+((col-c)*self.sc).square()+normal.square()
        return d2.masked_fill(~good,float('inf'))

    def __call__(self, xyz):
        xyz=xyz.reshape(-1,3).float().contiguous()
        total,h,w=self.images.shape
        mid=torch.empty(len(xyz),device=xyz.device,dtype=torch.int32)
        _nearest_frame[(triton.cdiv(len(xyz),128),)](xyz,self.inverse,self.mask,mid,len(xyz),
            total,h,w,1/self.sr,1/self.sc,self.r0,self.c0,self.radius_mm,BLOCK=128,
            enable_fp_fusion=False)
        mid=mid.long()
        dl,dr=self.center(xyz,mid-1),self.center(xyz,mid+1)
        neighbour=torch.where(dr<dl,mid+1,mid-1)
        weights=torch.zeros_like(xyz[:,0])
        intensities=torch.zeros_like(weights)
        exact_values=torch.zeros_like(weights)
        hits=torch.zeros_like(weights)
        for frames,available in ((mid,mid>=0),(neighbour,(mid>=0) & torch.isfinite(torch.minimum(dl,dr)))):
            ids=frames.clamp(0,total-1)
            row,col,normal=self.project(xyz,ids)
            r=torch.floor(row+.5).long()[:,None,None]+self.neighbour_offsets[None,:,None]
            c=torch.floor(col+.5).long()[:,None,None]+self.neighbour_offsets[None,None,:]
            rr,cc=r.clamp(0,h-1),c.clamp(0,w-1)
            valid=(available[:,None,None] & (rr==r) & (cc==c) & self.mask[ids[:,None,None],rr,cc].bool())
            distance=(((row[:,None,None]-r)*self.sr).square()+((col[:,None,None]-c)*self.sc).square()
                      +normal[:,None,None].square()).sqrt()
            # Rigid poses and their inverse are float32. Preserve the exact sample limit of IDW.
            exact=valid & (distance<3e-5)
            weight=valid.float()/distance.clamp_min(3e-5)
            gray=self.images[ids[:,None,None],rr,cc]
            weights+=weight.sum((1,2))
            intensities+=(gray*weight).sum((1,2))
            exact_values+=(gray*exact).sum((1,2))
            hits+=exact.sum((1,2))
        return torch.where(hits>0,exact_values/hits.clamp_min(1),intensities/weights.clamp_min(1e-10)), weights>0
