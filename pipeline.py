
import trimesh
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep3dfacerecon.networks import ReconNetWrapper
from deep3dfacerecon.bfm import ParametricFaceModel
from deep3dfacerecon.nvdiffrast import MeshRenderer

class Face3D():
    def __init__(self):
        self.face_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False).to("cuda")
        self.face_3d.load_state_dict(torch.load("/root/epoch_20.pth", map_location='cpu')['net_recon'])
        self.face_3d.eval()
        self.face_3d.cuda()
        self.face_model = ParametricFaceModel()
        self.face_model.to("cuda")
        center = 112.0
        focal = 1015.0
        fov = 2 * np.arctan(center / focal) * 180 / np.pi

        self.renderer = MeshRenderer(rasterize_fov=fov, znear=5.0, zfar=15.0, rasterize_size=int(2 * center))

        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
        ])

        self.to_pil = ToPILImage()

    @torch.no_grad()
    def __call__(self, i_s, i_t):
        i_s = self.transform(i_s).unsqueeze(dim=0).cuda()
        i_t = self.transform(i_t).unsqueeze(dim=0).cuda()
        c_s = self.face_3d(F.interpolate(i_s, size=224, mode='bilinear'))
        c_t = self.face_3d(F.interpolate(i_t, size=224, mode='bilinear'))
        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
        pred_vertex, pred_tex, pred_color, pred_lm = self.face_model.compute_for_render(c_fuse)
        pred_mask, pred_depth, pred_face = self.renderer(pred_vertex, self.face_model.face_buf, feat=pred_color)

        recon_shape = pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.face_model.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
        mesh.export("out.obj")

        pred_face = self.to_pil(pred_face[0])
        pred_mask = self.to_pil(pred_mask[0])
        pred_depth = self.to_pil(pred_depth[0])

        return pred_face, pred_mask, pred_depth


if __name__ == "__main__":
    face3d = Face3D()
    i_s = Image.open("/root/1.face.jpg")
    i_t = Image.open("/root/1.face.jpg")
    pred_face, pred_mask, pred_depth = face3d(i_s, i_t)
    pred_face.save("face.jpg")
    pred_mask.save("mask.jpg")
    pred_depth.save("depth.jpg")
