import torch
import trimesh
import numpy as np

def makeCoord():
    origin = np.zeros((1, 3, 3))
    axis = np.identity(3)[np.newaxis, :]
    vertices = np.concatenate([origin, axis], axis=-1)
    return vertices.reshape(-1, 3)

def makeGround(width=10, height=10, grid=100):
    x = np.linspace(-width//2, +width//2, grid)
    y = np.zeros_like(x)
    z_forward = np.ones_like(x) * height // 2
    z_backward = np.ones_like(x) * -height // 2
    forward = np.stack([x, y, z_forward], axis=-1)
    backward = np.stack([x, y, z_backward], axis=-1)

    z = np.linspace(-width//2, +width//2, grid)
    y = np.zeros_like(z)
    x_left = np.ones_like(z) * -width // 2
    x_right = np.ones_like(z) * width // 2
    left = np.stack([x_left, y, z], axis=-1)
    right = np.stack([x_right, y, z], axis=-1)

    vertices = np.zeros((grid*4, 3))
    vertices[0::4] = forward
    vertices[1::4] = backward
    vertices[2::4] = left
    vertices[3::4] = right
    return vertices


def applyMat(mat, vertices):
    h_vertices = np.concatenate([vertices, np.ones_like(vertices[..., [0]])], axis=-1)
    return ( mat @ h_vertices.T).T[..., :3]

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_face_normals(verts, faces):
    v0 = verts[faces[..., 0].long()]
    v1 = verts[faces[..., 1].long()]
    v2 = verts[faces[..., 2].long()]
    face_normals = safe_normalize(torch.cross(v0 - v1, v2 - v0))
    return face_normals

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals.transpose(1, 0) * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)

def create_sphere(level):
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=0.5, color=None)
    return sphere.vertices.astype("f4"), sphere.faces.astype("i4")

def unit_size(verts):
    vmin, vmax = np.min(verts, axis=0), np.max(verts, axis=0) # aabb
    scale = 2 / np.max(vmax - vmin).item()
    offset = -(vmax + vmin) / 2
    print(f"unit size the mesh: scale:{scale}, offset - {offset}")
    verts = verts + offset # Center mesh on origin
    verts = verts * scale
    return verts

def auto_uv(verts, faces):
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    # Trimesh needs a material to export uv coordinates and always creates a *.mtl file.
    # Alternatively, we can use the `export` helper function to export the mesh as obj.
    # xatlas.export("output.obj", sphere.vertices[vmapping], indices, uvs)
    return verts[vmapping].astype("f4"), indices.astype("i4"), uvs.astype("f4")

def trimesh_auto_uv(m:trimesh.Trimesh):
    assert isinstance(m, trimesh.Trimesh)
    m.vertices, m.faces, m.visual.uv = auto_uv(m.vertices, m.faces)
    return m.visual.uv


def trimesh_load(path, *args, **kwargs):
    _scene = trimesh.load(path, *args, **kwargs)
    if isinstance(_scene, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(_scene)
    else:
        scene = _scene
    return scene

def numpyfy_scene(scene: trimesh.Scene):
    """
        given a trimesh.Scene return the concated verts, faces, uvs, mat_per_face \n
        note that mat_per_face contains material id start from 1
    """

    geometries = list(scene.geometry.values())

    # ------------- merge meshes ------------------
    faces = []
    verts = []
    uvs = []
    mat_per_face = [[0]] # add a mat 0 to face 0
    
    _face_index_offset = 0
    for gid, g in enumerate(geometries):
        verts.append(g.vertices.copy())
        faces.append(g.faces.copy() + _face_index_offset)
        try:
            uvs.append(g.visual.uv.copy())
        except AttributeError as e:
            raise AttributeError("Not all meshes has uv")
    
        mat_per_face.append(np.ones_like(g.faces[..., 0]) * (gid + 1)) # mat id start from 1
        _face_index_offset += len(g.vertices)
    
    verts = np.concatenate(verts, axis=0).astype("f4")
    faces = np.concatenate(faces, axis=0).astype("i4")
    uvs = np.concatenate(uvs, axis=0).astype("f4")
    mat_per_face = np.concatenate(mat_per_face, axis=0).astype("i4")

    return verts, faces, uvs, mat_per_face