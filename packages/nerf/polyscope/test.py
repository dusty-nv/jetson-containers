import os
import time
import polyscope as ps
import numpy as np

# Start Xvfb for headless rendering
os.system("Xvfb :99 -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = ":99"

# Wait for Xvfb to initialize
time.sleep(3)

# Initialize polyscope
ps.init()

# Register a point cloud
my_points = np.random.randn(100, 3)
ps.register_point_cloud("my points", my_points)

# Register a mesh
num_verts = 100
num_faces = 200
verts = np.random.rand(num_verts, 3)
faces = np.random.randint(0, num_verts, size=(num_faces, 3))
ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)

print("Polyscope test completed successfully!")
