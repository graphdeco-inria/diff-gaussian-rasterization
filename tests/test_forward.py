import os
import math
import torch
import unittest
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

class TestGaussianRasterization(unittest.TestCase):
  def test_gaussian_rasterization(self):
    # init parameters
    device = "cuda"
    num_points = 1000

    # generate random gaussian parameters
    torch.manual_seed(42)
    means3D = torch.randn((num_points, 3), dtype=torch.float, device=device)  # [N, 3]
    colors = torch.rand((num_points, 3), dtype=torch.float, device=device)    # [N, 3] RGB in [0,1]
    opacities = torch.sigmoid(torch.randn((num_points, 1), dtype=torch.float, device=device))  # [N, 1]
    scales = torch.rand((num_points, 3), dtype=torch.float, device=device)    # [N, 3]
    rotations = torch.randn((num_points, 4), dtype=torch.float, device=device)
    rotations = rotations / rotations.norm(dim=1, keepdim=True)  # normalization

    # set rasterization parameters
    image_height, image_width = 512, 512
    fov_x, fov_y = 60.0, 45.0  # Field Angle (degree)

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=float(math.tan(fov_x * 0.5 * math.pi / 180)),  # Convert to radians and get tangent
        tanfovy=float(math.tan(fov_y * 0.5 * math.pi / 180)),
        bg=torch.tensor([0, 0, 0], dtype=torch.float, device=device),  # background color（black）
        scale_modifier=float(1.0),  # scale factor
        viewmatrix=torch.eye(4, dtype=torch.float, device=device),  # assume the view matrix is the identity matrix
        projmatrix=torch.eye(4, dtype=torch.float, device=device),  # assume the view matrix is the identity matrix
        sh_degree=0,  # 0 for not use sh
        campos=torch.zeros(3, dtype=torch.float, device=device),  # camera position
        prefiltered=False,
        debug=False
    )

    # create rasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    color, radii = rasterizer(
        means3D=means3D,
        means2D=None, # not used
        shs=None, # as raster_settings.sh_degree = 0, no Spherical Harmonics will affect the final render result
        opacities=opacities, # opacity for render if cover or not cover
        colors_precomp=colors, # set the color to avoid SH calculate
        scales=scales, # scale factor to control Conv3D gaussian
        rotations=rotations, # rotation for 3D=>2D
        cov3D_precomp=None
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_file_path = os.path.join(current_dir, 'data/forward_golden.pt')
    if not os.path.exists(out_file_path):
      if not torch.version.hip:
        print("[INFO] saving golden result with cuda.")
        torch.save({'color': color.cpu(), 'radii': radii.cpu()}, out_file_path)
      else:
        print("[ERROR] File not exist!")
        self.assertTrue(0)
    else:
      data = torch.load(out_file_path)
      reference_color = data['color']
      reference_radii = data['radii']
      self.assertTrue(torch.allclose(color.cpu(), reference_color))
      self.assertTrue(torch.allclose(radii.cpu(), reference_radii))

if __name__ == "__main__":
  unittest.main()
