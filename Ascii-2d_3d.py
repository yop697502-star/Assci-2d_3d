"""
ASCII 3D Renderer Library
Thư viện render 3D hoàn chỉnh sử dụng ASCII characters
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import time

# ============================================================================
# I. CORE 3D / Math & Camera
# ============================================================================
class Vec1:
    def __init__(self, x=0.0):
        self.x = x

    # --- Operators ---
    def __add__(self, o): return Vec1(self.x + o.x)
    def __sub__(self, o): return Vec1(self.x - o.x)
    def __mul__(self, s): return Vec1(self.x * s)
    def __truediv__(self, s): return Vec1(self.x / s)

    # --- Length & normalize ---
    def length(self):
        return abs(self.x)

    def normalize(self):
        l = self.length()
        return Vec1(self.x / l) if l > 0 else Vec1(0)

    # --- Helpers ---
    def to_array(self):
        return np.array([self.x])

    def __repr__(self):
        return f"Vec1({self.x})"
        
class Vec2:
    def __init__(self, x=0, y=0):
        self.x, self.y = x, y
    
    def __add__(self, o): return Vec2(self.x + o.x, self.y + o.y)
    def __sub__(self, o): return Vec2(self.x - o.x, self.y - o.y)
    def __mul__(self, s): return Vec2(self.x * s, self.y * s)
    def to_array(self): return np.array([self.x, self.y])

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    
    def __add__(self, o): return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, s): return Vec3(self.x * s, self.y * s, self.z * s)
    def __truediv__(self, s): return Vec3(self.x / s, self.y / s, self.z / s)
    
    def dot(self, o): return self.x * o.x + self.y * o.y + self.z * o.z
    def cross(self, o):
        return Vec3(self.y * o.z - self.z * o.y,
                   self.z * o.x - self.x * o.z,
                   self.x * o.y - self.y * o.x)
    
    def length(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        l = self.length()
        return self / l if l > 0 else Vec3()
    
    def to_array(self): return np.array([self.x, self.y, self.z, 1.0])

class Vec4:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x, self.y, self.z, self.w = x, y, z, w
    
    def to_vec3(self):
        if self.w != 0:
            return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)
        return Vec3(self.x, self.y, self.z)
    
    def to_array(self): return np.array([self.x, self.y, self.z, self.w])

class Mat4:
    def __init__(self, data=None):
        self.m = data if data is not None else np.eye(4)
    
    @staticmethod
    def identity():
        return Mat4()
    
    @staticmethod
    def translation(x, y, z):
        m = np.eye(4)
        m[0:3, 3] = [x, y, z]
        return Mat4(m)
    
    @staticmethod
    def rotation_x(angle):
        c, s = math.cos(angle), math.sin(angle)
        m = np.eye(4)
        m[1, 1], m[1, 2] = c, -s
        m[2, 1], m[2, 2] = s, c
        return Mat4(m)
    
    @staticmethod
    def rotation_y(angle):
        c, s = math.cos(angle), math.sin(angle)
        m = np.eye(4)
        m[0, 0], m[0, 2] = c, s
        m[2, 0], m[2, 2] = -s, c
        return Mat4(m)
    
    @staticmethod
    def rotation_z(angle):
        c, s = math.cos(angle), math.sin(angle)
        m = np.eye(4)
        m[0, 0], m[0, 1] = c, -s
        m[1, 0], m[1, 1] = s, c
        return Mat4(m)
    
    @staticmethod
    def scale(x, y, z):
        m = np.eye(4)
        m[0, 0], m[1, 1], m[2, 2] = x, y, z
        return Mat4(m)
    
    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3):
        z = (eye - target).normalize()
        x = up.cross(z).normalize()
        y = z.cross(x)
        
        m = np.eye(4)
        m[0, :3] = [x.x, x.y, x.z]
        m[1, :3] = [y.x, y.y, y.z]
        m[2, :3] = [z.x, z.y, z.z]
        m[0:3, 3] = [-x.dot(eye), -y.dot(eye), -z.dot(eye)]
        return Mat4(m)
    
    @staticmethod
    def perspective(fov, aspect, near, far):
        f = 1.0 / math.tan(fov / 2)
        m = np.zeros((4, 4))
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1
        return Mat4(m)
    
    @staticmethod
    def orthographic(left, right, bottom, top, near, far):
        m = np.eye(4)
        m[0, 0] = 2 / (right - left)
        m[1, 1] = 2 / (top - bottom)
        m[2, 2] = -2 / (far - near)
        m[0, 3] = -(right + left) / (right - left)
        m[1, 3] = -(top + bottom) / (top - bottom)
        m[2, 3] = -(far + near) / (far - near)
        return Mat4(m)
    
    def __matmul__(self, other):
        if isinstance(other, Mat4):
            return Mat4(self.m @ other.m)
        elif isinstance(other, Vec3):
            v = other.to_array()
            result = self.m @ v
            return Vec4(*result)
        return NotImplemented
    
    def transform_point(self, v: Vec3) -> Vec3:
        result = self @ v
        return result.to_vec3()
    
    def transform_direction(self, v: Vec3) -> Vec3:
        arr = np.array([v.x, v.y, v.z, 0.0])
        result = self.m @ arr
        return Vec3(result[0], result[1], result[2])

class ProjectionType(Enum):
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2

@dataclass
class Camera:
    position: Vec3 = Vec3(0, 0, 5)
    target: Vec3 = Vec3(0, 0, 0)
    up: Vec3 = Vec3(0, 1, 0)
    fov: float = math.radians(60)
    aspect: float = 1.0
    near: float = 0.1
    far: float = 100.0
    projection_type: ProjectionType = ProjectionType.PERSPECTIVE
    
    def get_view_matrix(self):
        return Mat4.look_at(self.position, self.target, self.up)
    
    def get_projection_matrix(self):
        if self.projection_type == ProjectionType.PERSPECTIVE:
            return Mat4.perspective(self.fov, self.aspect, self.near, self.far)
        else:
            size = 5.0
            return Mat4.orthographic(-size * self.aspect, size * self.aspect,
                                    -size, size, self.near, self.far)

# ============================================================================
# II. Mesh / Geometry
# ============================================================================

@dataclass
class Vertex:
    position: Vec3
    normal: Vec3 = Vec3(0, 1, 0)
    uv: Vec2 = Vec2(0, 0)
    color: Vec3 = Vec3(1, 1, 1)

@dataclass
class Triangle:
    v0: Vertex
    v1: Vertex
    v2: Vertex
    
    def get_normal(self):
        e1 = self.v1.position - self.v0.position
        e2 = self.v2.position - self.v0.position
        return e1.cross(e2).normalize()

class Mesh:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.triangles: List[Triangle] = []
        self.transform = Mat4.identity()
    
    @staticmethod
    def create_cube(size=1.0):
        mesh = Mesh()
        s = size / 2
        
        # Define 8 vertices of cube
        positions = [
            Vec3(-s, -s, -s), Vec3(s, -s, -s), Vec3(s, s, -s), Vec3(-s, s, -s),
            Vec3(-s, -s, s), Vec3(s, -s, s), Vec3(s, s, s), Vec3(-s, s, s)
        ]
        
        # 12 triangles (2 per face)
        indices = [
            0,1,2, 0,2,3,  # front
            1,5,6, 1,6,2,  # right
            5,4,7, 5,7,6,  # back
            4,0,3, 4,3,7,  # left
            3,2,6, 3,6,7,  # top
            4,5,1, 4,1,0   # bottom
        ]
        
        for i in range(0, len(indices), 3):
            v0 = Vertex(positions[indices[i]])
            v1 = Vertex(positions[indices[i+1]])
            v2 = Vertex(positions[indices[i+2]])
            mesh.triangles.append(Triangle(v0, v1, v2))
        
        mesh.calculate_normals()
        return mesh
    
    @staticmethod
    def create_sphere(radius=1.0, segments=16):
        mesh = Mesh()
        
        for lat in range(segments + 1):
            theta = lat * math.pi / segments
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            for lon in range(segments + 1):
                phi = lon * 2 * math.pi / segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                
                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta
                
                v = Vertex(
                    position=Vec3(x * radius, y * radius, z * radius),
                    normal=Vec3(x, y, z).normalize(),
                    uv=Vec2(lon / segments, lat / segments)
                )
                mesh.vertices.append(v)
        
        # Create triangles
        for lat in range(segments):
            for lon in range(segments):
                first = lat * (segments + 1) + lon
                second = first + segments + 1
                
                v0 = mesh.vertices[first]
                v1 = mesh.vertices[second]
                v2 = mesh.vertices[first + 1]
                v3 = mesh.vertices[second + 1]
                
                mesh.triangles.append(Triangle(v0, v1, v2))
                mesh.triangles.append(Triangle(v2, v1, v3))
        
        return mesh
    
    def calculate_normals(self):
        """Calculate smooth normals for vertices"""
        for tri in self.triangles:
            normal = tri.get_normal()
            tri.v0.normal = normal
            tri.v1.normal = normal
            tri.v2.normal = normal
    
    @staticmethod
    def load_obj(filename):
        """Basic OBJ loader"""
        mesh = Mesh()
        vertices = []
        normals = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':
                        vertices.append(Vec3(float(parts[1]), 
                                           float(parts[2]), 
                                           float(parts[3])))
                    elif parts[0] == 'vn':
                        normals.append(Vec3(float(parts[1]), 
                                          float(parts[2]), 
                                          float(parts[3])))
                    elif parts[0] == 'f':
                        # Parse face (supports v, v/vt, v/vt/vn, v//vn)
                        face_verts = []
                        for i in range(1, 4):
                            idx = parts[i].split('/')[0]
                            v_idx = int(idx) - 1
                            face_verts.append(Vertex(vertices[v_idx]))
                        
                        mesh.triangles.append(Triangle(*face_verts))
            
            mesh.calculate_normals()
        except FileNotFoundError:
            print(f"File {filename} not found, creating default cube")
            return Mesh.create_cube()
        
        return mesh

# ============================================================================
# III. LIGHTING & SHADING
# ============================================================================

class Light:
    def __init__(self, position=Vec3(0, 10, 0), color=Vec3(1, 1, 1), intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class DirectionalLight:
    def __init__(self, direction=Vec3(0, -1, 0), color=Vec3(1, 1, 1), intensity=1.0):
        self.direction = direction.normalize()
        self.color = color
        self.intensity = intensity

class SpotLight:
    def __init__(self, position=Vec3(0, 5, 0), direction=Vec3(0, -1, 0), 
                 color=Vec3(1, 1, 1), intensity=1.0, cutoff=0.9):
        self.position = position
        self.direction = direction.normalize()
        self.color = color
        self.intensity = intensity
        self.cutoff = cutoff

class Shader:
    # ASCII shading levels
    SHADING_CHARS = [' ', '░', '▒', '▓', '█']
    
    @staticmethod
    def lambert(normal: Vec3, light_dir: Vec3, light_color: Vec3, intensity: float):
        """Diffuse (Lambert) shading"""
        ndotl = max(0, normal.dot(light_dir))
        return light_color * (ndotl * intensity)
    
    @staticmethod
    def phong(normal: Vec3, light_dir: Vec3, view_dir: Vec3, 
              light_color: Vec3, intensity: float, shininess=32):
        """Specular (Phong) shading"""
        reflect_dir = (normal * (2 * normal.dot(light_dir)) - light_dir).normalize()
        spec = max(0, view_dir.dot(reflect_dir)) ** shininess
        return light_color * (spec * intensity)
    
    @staticmethod
    def get_ascii_char(brightness: float, depth_fade=1.0):
        """Convert brightness to ASCII character"""
        brightness = max(0, min(1, brightness * depth_fade))
        idx = int(brightness * (len(Shader.SHADING_CHARS) - 1))
        return Shader.SHADING_CHARS[idx]

# ============================================================================
# IV. SHADOW SYSTEM
# ============================================================================

class ShadowMap:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.depth_buffer = np.full((height, width), float('inf'))
    
    def clear(self):
        self.depth_buffer.fill(float('inf'))
    
    def sample(self, x: int, y: int, depth: float, bias=0.005):
        """Check if point is in shadow"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return depth > self.depth_buffer[y, x] + bias
        return False
    
    def sample_pcf(self, x: float, y: float, depth: float, radius=2):
        """Soft shadow with PCF (Percentage Closer Filtering)"""
        shadow = 0.0
        samples = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ix, iy = int(x + dx), int(y + dy)
                if self.sample(ix, iy, depth):
                    shadow += 1.0
                samples += 1
        
        return shadow / samples if samples > 0 else 0.0

# ============================================================================
# V. TEXTURE & EFFECTS
# ============================================================================

class Texture:
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.data = np.ones((height, width, 3))
    
    @staticmethod
    def create_checker(size=64):
        tex = Texture(size, size)
        for y in range(size):
            for x in range(size):
                checker = ((x // 8) + (y // 8)) % 2
                tex.data[y, x] = [checker, checker, checker]
        return tex
    
    def sample(self, u: float, v: float):
        """Sample texture at UV coordinates"""
        u = u % 1.0
        v = v % 1.0
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        return Vec3(*self.data[y, x])

# ============================================================================
# VI. RASTERIZER & RENDERER
# ============================================================================

class FrameBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.color_buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.depth_buffer = np.full((height, width), float('inf'))
        self.brightness_buffer = np.zeros((height, width))
    
    def clear(self, char=' '):
        self.color_buffer = [[char for _ in range(self.width)] 
                            for _ in range(self.height)]
        self.depth_buffer.fill(float('inf'))
        self.brightness_buffer.fill(0)
    
    def set_pixel(self, x: int, y: int, char: str, depth: float, brightness=1.0):
        if 0 <= x < self.width and 0 <= y < self.height:
            if depth < self.depth_buffer[y, x]:
                self.color_buffer[y][x] = char
                self.depth_buffer[y, x] = depth
                self.brightness_buffer[y, x] = brightness
    
    def to_string(self):
        return '\n'.join(''.join(row) for row in self.color_buffer)

class Renderer:
    def __init__(self, width=80, height=40):
        self.width = width
        self.height = height
        self.framebuffer = FrameBuffer(width, height)
        self.camera = Camera(aspect=width / height)
        self.lights: List[Light] = []
        self.ambient = Vec3(0.2, 0.2, 0.2)
        self.wireframe_mode = False
        self.backface_culling = True
        self.enable_shadows = False
        self.shadow_map = ShadowMap(256, 256)
        self.enable_fog = False
        self.fog_density = 0.05
        self.show_fps = True
        self.frame_time = 0
        
    def add_light(self, light):
        self.lights.append(light)
    
    def clear(self):
        self.framebuffer.clear()
    
    def project_vertex(self, vertex: Vertex, mvp: Mat4):
        """Transform vertex through MVP matrix"""
        pos_clip = mvp @ vertex.position
        
        # Perspective divide
        if pos_clip.w != 0:
            ndc_x = pos_clip.x / pos_clip.w
            ndc_y = pos_clip.y / pos_clip.w
            ndc_z = pos_clip.z / pos_clip.w
        else:
            ndc_x, ndc_y, ndc_z = 0, 0, 0
        
        # Screen space mapping
        screen_x = (ndc_x + 1) * 0.5 * self.width
        screen_y = (1 - ndc_y) * 0.5 * self.height  # Flip Y
        
        return screen_x, screen_y, ndc_z, pos_clip.w
    
    def draw_line(self, x0, y0, x1, y1, char='█'):
        """Bresenham line drawing"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.width and 0 <= y0 < self.height:
                self.framebuffer.set_pixel(int(x0), int(y0), char, 0)
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def rasterize_triangle(self, tri: Triangle, mvp: Mat4, model: Mat4):
        """Rasterize a triangle with full shading"""
        # Project vertices
        v0_proj = self.project_vertex(tri.v0, mvp)
        v1_proj = self.project_vertex(tri.v1, mvp)
        v2_proj = self.project_vertex(tri.v2, mvp)
        
        # Extract screen coordinates
        x0, y0, z0, w0 = v0_proj
        x1, y1, z1, w1 = v1_proj
        x2, y2, z2, w2 = v2_proj
        
        # Wireframe mode
        if self.wireframe_mode:
            self.draw_line(x0, y0, x1, y1, '█')
            self.draw_line(x1, y1, x2, y2, '█')
            self.draw_line(x2, y2, x0, y0, '█')
            return
        
        # Backface culling
        if self.backface_culling:
            edge1 = Vec2(x1 - x0, y1 - y0)
            edge2 = Vec2(x2 - x0, y2 - y0)
            cross = edge1.x * edge2.y - edge1.y * edge2.x
            if cross <= 0:
                return
        
        # Bounding box
        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(self.width - 1, int(max(x0, x1, x2)))
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(self.height - 1, int(max(y0, y1, y2)))
        
        # Transform normals to world space
        n0 = model.transform_direction(tri.v0.normal).normalize()
        n1 = model.transform_direction(tri.v1.normal).normalize()
        n2 = model.transform_direction(tri.v2.normal).normalize()
        
        # World positions
        p0 = model.transform_point(tri.v0.position)
        p1 = model.transform_point(tri.v1.position)
        p2 = model.transform_point(tri.v2.position)
        
        # Rasterize
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Barycentric coordinates
                v0 = Vec2(x2 - x0, y2 - y0)
                v1 = Vec2(x1 - x0, y1 - y0)
                v2 = Vec2(x - x0, y - y0)
                
                d00 = v0.x * v0.x + v0.y * v0.y
                d01 = v0.x * v1.x + v0.y * v1.y
                d11 = v1.x * v1.x + v1.y * v1.y
                d20 = v2.x * v0.x + v2.y * v0.y
                d21 = v2.x * v1.x + v2.y * v1.y
                
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-8:
                    continue
                
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1 - v - w
                
                # Inside triangle?
                if u >= 0 and v >= 0 and w >= 0:
                    # Interpolate depth
                    depth = u * z0 + v * z1 + w * z2
                    
                    # Depth test
                    if depth < self.framebuffer.depth_buffer[y, x]:
                        # Interpolate normal (Gouraud shading)
                        normal = (n0 * u + n1 * v + n2 * w).normalize()
                        
                        # Interpolate world position
                        world_pos = Vec3(
                            p0.x * u + p1.x * v + p2.x * w,
                            p0.y * u + p1.y * v + p2.y * w,
                            p0.z * u + p1.z * v + p2.z * w
                        )
                        
                        # Calculate lighting
                        view_dir = (self.camera.position - world_pos).normalize()
                        total_light = self.ambient
                        
                        for light in self.lights:
                            if isinstance(light, Light):
                                light_dir = (light.position - world_pos).normalize()
                                diffuse = Shader.lambert(normal, light_dir, 
                                                        light.color, light.intensity)
                                specular = Shader.phong(normal, light_dir, view_dir,
                                                       light.color, light.intensity)
                                total_light = total_light + diffuse + specular * 0.5
                        
                        # Calculate brightness
                        brightness = (total_light.x + total_light.y + total_light.z) / 3
                        
                        # Fog effect
                        if self.enable_fog:
                            dist = (world_pos - self.camera.position).length()
                            fog_factor = math.exp(-self.fog_density * dist)
                            brightness *= fog_factor
                        
                        # Get ASCII character
                        char = Shader.get_ascii_char(brightness)
                        self.framebuffer.set_pixel(x, y, char, depth, brightness)
    
    def render(self, mesh: Mesh):
        """Render a mesh"""
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        model = mesh.transform
        mvp = proj @ view @ model
        
        for tri in mesh.triangles:
            self.rasterize_triangle(tri, mvp, model)
    
    def get_frame(self):
        """Get rendered frame as string"""
        output = self.framebuffer.to_string()
        
        if self.show_fps:
            fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
            output = f"FPS: {fps:.1f}\n" + output
        
        return output

# ============================================================================
# VII. SCENE GRAPH
# ============================================================================

class SceneNode:
    def __init__(self, name="Node"):
        self.name = name
        self.transform = Mat4.identity()
        self.children: List[SceneNode] = []
        self.mesh: Optional[Mesh] = None
        self.parent: Optional[SceneNode] = None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    
    def get_world_transform(self):
        if self.parent:
            return self.parent.get_world_transform() @ self.transform
        return self.transform
    
    def render(self, renderer: Renderer):
        if self.mesh:
            self.mesh.transform = self.get_world_transform()
            renderer.render(self.mesh)
        
        for child in self.children:
            child.render(renderer)         
"""
ASCII 3D Dual Pipeline (Full)
- Fixed Function Pipeline (ASCII 4-level) + Core Profile (RGB)
- Dùng lại Mesh, Camera, Light, Shader, FrameBuffer
"""

from copy import deepcopy
import math
from typing import List, Optional

ASCII_CHARS = [' ', '░', '▒', '█']

def brightness_to_ascii(b):
    """Convert brightness (0..1) to 4-level ASCII char"""
    idx = max(0, min(3, int(b * 4)))
    return ASCII_CHARS[idx]

# ------------------------------------------------------------------------
# I. Matrix Stack for Fixed Function
# ------------------------------------------------------------------------
class MatrixStack:
    """Mimic OpenGL fixed-function matrix stack"""
    def __init__(self):
        self.stack = [Mat4.identity()]
    
    def push(self):
        self.stack.append(deepcopy(self.stack[-1]))
    
    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise RuntimeError("Matrix stack underflow")
    
    def load_identity(self):
        self.stack[-1] = Mat4.identity()
    
    def multiply(self, mat: Mat4):
        self.stack[-1] = self.stack[-1] @ mat
    
    def current(self):
        return self.stack[-1]

# ------------------------------------------------------------------------
# II. Fixed Function Pipeline
# ------------------------------------------------------------------------
class FixedFunctionPipeline:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer
        self.model_stack = MatrixStack()
        self.view_stack = MatrixStack()
        self.proj_stack = MatrixStack()
        self.lights: List[Light] = []
        self.ambient = 0.2  # brightness only
    
    def set_camera(self, camera: Camera):
        self.view_stack.load_identity()
        self.view_stack.multiply(camera.get_view_matrix())
        self.proj_stack.load_identity()
        self.proj_stack.multiply(camera.get_projection_matrix())
    
    def push_matrix(self): self.model_stack.push()
    def pop_matrix(self): self.model_stack.pop()
    def load_identity(self): self.model_stack.load_identity()
    def translate(self, x, y, z): self.model_stack.multiply(Mat4.translation(x,y,z))
    def rotate_x(self, angle): self.model_stack.multiply(Mat4.rotation_x(angle))
    def rotate_y(self, angle): self.model_stack.multiply(Mat4.rotation_y(angle))
    def rotate_z(self, angle): self.model_stack.multiply(Mat4.rotation_z(angle))
    def scale(self, x, y, z): self.model_stack.multiply(Mat4.scale(x,y,z))
    def add_light(self, light: Light): self.lights.append(light)
    
    def render_mesh(self, mesh: Mesh):
        """Render mesh using fixed-function ASCII 4-level shading"""
        mvp = self.proj_stack.current() @ self.view_stack.current() @ self.model_stack.current()
        model = self.model_stack.current()
        
        for tri in mesh.triangles:
            # Transform normals & positions
            n0 = model.transform_direction(tri.v0.normal).normalize()
            n1 = model.transform_direction(tri.v1.normal).normalize()
            n2 = model.transform_direction(tri.v2.normal).normalize()
            p0 = model.transform_point(tri.v0.position)
            p1 = model.transform_point(tri.v1.position)
            p2 = model.transform_point(tri.v2.position)
            
            # Calculate per-vertex brightness (Lambert + simple specular)
            for v, n, p in zip([tri.v0, tri.v1, tri.v2], [n0,n1,n2], [p0,p1,p2]):
                view_dir = (self.renderer.camera.position - p).normalize()
                brightness = self.ambient
                for light in self.lights:
                    light_dir = (light.position - p).normalize()
                    diffuse = max(0, n.dot(light_dir)) * light.intensity
                    # simple ASCII specular
                    reflect_dir = (n * (2 * n.dot(light_dir)) - light_dir).normalize()
                    spec = max(0, view_dir.dot(reflect_dir)) ** 16
                    brightness += diffuse + 0.3 * spec
                v.color = max(0, min(1, brightness))  # store brightness only
            
            # Rasterize triangle in ASCII
            self.renderer.rasterize_triangle_ascii(tri, mvp, model)

# ------------------------------------------------------------------------
# III. Core Profile Pipeline (Wrap Existing Renderer)
# ------------------------------------------------------------------------
class CoreProfilePipeline:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer
    
    def render_mesh(self, mesh: Mesh):
        self.renderer.render(mesh)

# ------------------------------------------------------------------------
# IV. Renderer modification: rasterize_triangle_ascii
# ------------------------------------------------------------------------
def rasterize_triangle_ascii(self, tri: Triangle, mvp: Mat4, model: Mat4):
    """Rasterize triangle using brightness only (ASCII 4-level)"""
    # Project vertices
    def proj(v): 
        x, y, z, w = self.project_vertex(v, mvp)
        return Vec2(x, y), z, v.color  # position, depth, brightness

    v0, z0, b0 = proj(tri.v0)
    v1, z1, b1 = proj(tri.v1)
    v2, z2, b2 = proj(tri.v2)
    
    # Bounding box
    min_x = max(0, int(min(v0.x, v1.x, v2.x)))
    max_x = min(self.width-1, int(max(v0.x, v1.x, v2.x)))
    min_y = max(0, int(min(v0.y, v1.y, v2.y)))
    max_y = min(self.height-1, int(max(v0.y, v1.y, v2.y)))
    
    # Barycentric rasterization
    def edge(a,b,p): return (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x)
    for y in range(min_y, max_y+1):
        for x in range(min_x, max_x+1):
            p = Vec2(x+0.5, y+0.5)
            w0 = edge(v1,v2,p)
            w1 = edge(v2,v0,p)
            w2 = edge(v0,v1,p)
            if w0>=0 and w1>=0 and w2>=0:
                area = edge(v0,v1,v2)
                u, v, w = w0/area, w1/area, w2/area
                depth = u*z0 + v*z1 + w*z2
                brightness = u*b0 + v*b1 + w*b2
                char = brightness_to_ascii(brightness)
                self.framebuffer.set_pixel(x, y, char, depth, brightness)

# Patch Renderer
Renderer.rasterize_triangle_ascii = rasterize_triangle_ascii

# ------------------------------------------------------------------------
# V. Unified ASCII 3D Dual Pipeline
# ------------------------------------------------------------------------
class ASCIIPipeline:
    def __init__(self, width=80, height=40):
        self.renderer = Renderer(width, height)
        self.fixed_func = FixedFunctionPipeline(self.renderer)
        self.core_profile = CoreProfilePipeline(self.renderer)
        self.use_fixed_function = True
    
    def set_pipeline(self, fixed_function=True):
        self.use_fixed_function = fixed_function
    
    def set_camera(self, camera: Camera):
        self.renderer.camera = camera
        self.fixed_func.set_camera(camera)
    
    def add_light(self, light):
        self.renderer.add_light(light)
        self.fixed_func.add_light(light)
    
    def clear(self):
        self.renderer.clear()
    
    def render_mesh(self, mesh: Mesh):
        if self.use_fixed_function:
            self.fixed_func.render_mesh(mesh)
        else:
            self.core_profile.render_mesh(mesh)
    
    def get_frame(self):
        return self.renderer.get_frame()  
# ============================================================================
# Tessellator ASCII
# ============================================================================
class Tessellator:
    @staticmethod
    def subdivide_triangle(tri: Triangle, level=1) -> List[Triangle]:
        """CPU tessellation: chia tam giác level lần, interpolate brightness"""
        if level <= 0:
            return [tri]
        
        v0, v1, v2 = tri.v0, tri.v1, tri.v2
        
        def midpoint(a: Vertex, b: Vertex) -> Vertex:
            pos = (a.position + b.position) * 0.5
            norm = (a.normal + b.normal).normalize()
            brightness = (a.color.x + b.color.x) / 2  # brightness stored in color.x
            return Vertex(pos, norm, Vec2(0,0), Vec3(brightness,0,0))
        
        m01 = midpoint(v0, v1)
        m12 = midpoint(v1, v2)
        m20 = midpoint(v2, v0)
        
        tris = [
            Triangle(v0, m01, m20),
            Triangle(m01, v1, m12),
            Triangle(m20, m12, v2),
            Triangle(m01, m12, m20)
        ]
        
        result = []
        for t in tris:
            result.extend(Tessellator.subdivide_triangle(t, level-1))
        return result

# ============================================================================
# Geometry Shader CPU ASCII
# ============================================================================
class GeometryShaderCPU:
    @staticmethod
    def extrude_triangle(tri: Triangle, distance=0.1) -> List[Triangle]:
        """Extrude triangle along normal, keep brightness"""
        n = tri.get_normal()
        v0 = Vertex(tri.v0.position + n * distance, tri.v0.normal, Vec2(0,0), tri.v0.color)
        v1 = Vertex(tri.v1.position + n * distance, tri.v1.normal, Vec2(0,0), tri.v1.color)
        v2 = Vertex(tri.v2.position + n * distance, tri.v2.normal, Vec2(0,0), tri.v2.color)
        return [tri, Triangle(v0, v1, v2)]

# ============================================================================
# Instancer ASCII
# ============================================================================
class Instancer:
    @staticmethod
    def render_instances(renderer: Renderer, mesh: Mesh, transforms: List[Mat4]):
        for t in transforms:
            mesh_copy = Mesh()
            mesh_copy.vertices = [Vertex(v.position, v.normal, v.uv, v.color) 
                                  for v in mesh.vertices]
            tri_map = {id(v): mesh_copy.vertices[i] for i,v in enumerate(mesh.vertices)}
            mesh_copy.triangles = [Triangle(tri_map[id(tri.v0)],
                                            tri_map[id(tri.v1)],
                                            tri_map[id(tri.v2)]) 
                                   for tri in mesh.triangles]
            mesh_copy.transform = t
            renderer.render(mesh_copy)

# ============================================================================
# GBuffer ASCII
# ============================================================================
class GBuffer:
    """ASCII-only GBuffer: brightness, normal, depth"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.brightness = np.zeros((height, width))
        self.normal = [[Vec3(0,0,1) for _ in range(width)] for _ in range(height)]
        self.depth = np.full((height, width), float('inf'))
    
    def clear(self):
        self.brightness.fill(0)
        for y in range(self.height):
            for x in range(self.width):
                self.normal[y][x] = Vec3(0,0,1)
                self.depth[y,x] = float('inf')   
# ============================================================================
# ASCII Shader & Lighting Module
# ============================================================================

class ASCIILight:
    """Point light for ASCII shading"""
    def __init__(self, position: Vec3, intensity: float=1.0):
        self.position = position
        self.intensity = intensity

class DirectionalASCIILight:
    """Directional light for ASCII shading"""
    def __init__(self, direction: Vec3, intensity: float=1.0):
        self.direction = direction.normalize()
        self.intensity = intensity

class SpotASCIILight:
    """Spot light for ASCII shading"""
    def __init__(self, position: Vec3, direction: Vec3, 
                 cutoff: float=0.9, intensity: float=1.0):
        self.position = position
        self.direction = direction.normalize()
        self.cutoff = cutoff  # cos(angle)
        self.intensity = intensity

class ASCIIShader:
    """Compute brightness per vertex or per pixel"""
    
    SHADING_CHARS = [' ', '░', '▒', '▓', '█']  # 5 levels
    
    @staticmethod
    def lambert(normal: Vec3, light_dir: Vec3, intensity: float) -> float:
        """Diffuse Lambertian brightness"""
        ndotl = max(0, normal.dot(light_dir))
        return ndotl * intensity
    
    @staticmethod
    def phong(normal: Vec3, light_dir: Vec3, view_dir: Vec3, intensity: float, shininess=16) -> float:
        """Specular Phong brightness"""
        reflect = (normal * (2 * normal.dot(light_dir)) - light_dir).normalize()
        spec = max(0, view_dir.dot(reflect)) ** shininess
        return spec * intensity
    
    @staticmethod
    def apply_lights(world_pos: Vec3, normal: Vec3, lights: list, view_pos: Vec3,
                     ambient: float=0.2, fog_density: float=0.0) -> float:
        """Compute final brightness at a point with multiple lights"""
        brightness = ambient
        view_dir = (view_pos - world_pos).normalize()
        
        for light in lights:
            if isinstance(light, ASCIILight):
                ldir = (light.position - world_pos).normalize()
                brightness += ASCIIShader.lambert(normal, ldir, light.intensity)
                brightness += ASCIIShader.phong(normal, ldir, view_dir, light.intensity, 16) * 0.3
            elif isinstance(light, DirectionalASCIILight):
                ldir = -light.direction
                brightness += ASCIIShader.lambert(normal, ldir, light.intensity)
                brightness += ASCIIShader.phong(normal, ldir, view_dir, light.intensity, 16) * 0.3
            elif isinstance(light, SpotASCIILight):
                ldir = (light.position - world_pos).normalize()
                spot_factor = ldir.dot(-light.direction)
                if spot_factor > light.cutoff:
                    brightness += ASCIIShader.lambert(normal, ldir, light.intensity * spot_factor)
                    brightness += ASCIIShader.phong(normal, ldir, view_dir, light.intensity * spot_factor, 16) * 0.3
        
        # Fog effect
        if fog_density > 0.0:
            dist = (world_pos - view_pos).length()
            fog_factor = math.exp(-fog_density * dist)
            brightness *= fog_factor
        
        return max(0.0, min(1.0, brightness))
    
    @staticmethod
    def brightness_to_ascii(brightness: float) -> str:
        """Convert brightness (0..1) to ASCII char"""
        idx = int(brightness * (len(ASCIIShader.SHADING_CHARS) - 1))
        return ASCIIShader.SHADING_CHARS[idx]

# ============================================================================
# Shadow Map for ASCII
# ============================================================================

class ASCIIShadowMap:
    """Simple shadow map for ASCII shading (depth buffer)"""
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        self.depth = np.full((height, width), float('inf'))
    
    def clear(self):
        self.depth.fill(float('inf'))
    
    def in_shadow(self, x: int, y: int, depth: float, bias=0.005) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return depth > self.depth[y, x] + bias
        return False
    
    def sample_pcf(self, x: float, y: float, depth: float, radius=2) -> float:
        """Soft shadow: return 0..1 fraction in shadow"""
        shadow = 0.0
        samples = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                ix, iy = int(x + dx), int(y + dy)
                if self.in_shadow(ix, iy, depth):
                    shadow += 1.0
                samples += 1
        return shadow / samples if samples > 0 else 0.0        

#!/usr/bin/env python3
"""
ascii_gdi_full_v4.py
Full ASCII GDI-like engine (text-only) v4
"""

from math import ceil, floor, sqrt, sin, cos, radians
from collections import namedtuple, deque
import copy
from PIL import Image

Rect = namedtuple("Rect", ["x", "y", "w", "h"])

def clamp(v, lo, hi): return max(lo, min(hi, v))
def in_rect(x, y, r): return r.x <= x < r.x + r.w and r.y <= y < r.y + r.h

# ---------------- Canvas ----------------
class Canvas:
    def __init__(self, w, h, bg=' '):
        self.w, self.h = int(w), int(h)
        self.bg = bg
        self.pixels = [[bg for _ in range(w)] for __ in range(h)]

    def clear(self, ch=None):
        c = ch if ch is not None else self.bg
        self.pixels = [[c for _ in range(self.w)] for __ in range(self.h)]

    def in_bounds(self, x, y): return 0 <= x < self.w and 0 <= y < self.h
    def get(self, x, y): return self.pixels[y][x] if self.in_bounds(x, y) else self.bg
    def set(self, x, y, ch):
        if self.in_bounds(x, y): self.pixels[y][x] = ch

    def as_lines(self): return [''.join(row) for row in self.pixels]

    def save_text(self, path):
        with open(path, 'w', encoding='utf8') as f:
            for line in self.as_lines(): f.write(line + '\n')

    def snapshot(self): return [row[:] for row in self.pixels]
    def restore_snapshot(self, snap): self.pixels = [row[:] for row in snap]

    # Layer alpha blend
    def blend_pixel(self, x, y, ch, alpha=1.0):
        if not self.in_bounds(x, y): return
        bg_ch = self.get(x, y)
        gradient = [' ', '░', '▒', '▓', '█']
        val_bg = gradient.index(bg_ch) / (len(gradient) - 1) if bg_ch in gradient else 0
        val_fg = gradient.index(ch) / (len(gradient) - 1) if ch in gradient else 1
        val = val_fg * alpha + val_bg * (1 - alpha)
        idx = int(clamp(val, 0, 1) * (len(gradient) - 1))
        self.set(x, y, gradient[idx])

# ---------------- GDI Objects ----------------
class Pen:
    def __init__(self, ch='*', width=1, style='solid'):
        self.ch = ch
        self.width = max(1, int(width))
        self.style = style
    def dash_pattern(self):
        if self.style=='dash': return [1,1,0,0]
        if self.style=='dot': return [1,0,1,0]
        return None

class Brush:
    def __init__(self, ch='#', pattern=None, gradient=None, transparent=False):
        self.ch = ch
        self.pattern = pattern
        self.gradient = gradient
        self.transparent = transparent
    def get_char(self, x=0, y=0, val=None):
        if self.pattern:
            h, w = len(self.pattern), len(self.pattern[0])
            return self.pattern[y%h][x%w]
        if self.gradient and val is not None:
            idx = int(clamp(val, 0, 1) * (len(self.gradient)-1))
            return self.gradient[idx]
        return self.ch

class Font:
    def __init__(self, size=1, bold=False, italic=False, underline=False, strike=False):
        self.size = int(size)
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.strike = strike

class Region:
    def __init__(self, rects=None):
        self.rects = [Rect(*r) for r in (rects or [])]
    def add_rect(self, x, y, w, h): self.rects.append(Rect(x, y, w, h))
    def contains(self, x, y):
        return any(in_rect(x, y, r) for r in self.rects)
    def intersect(self, other):
        out = Region()
        for a in self.rects:
            for b in other.rects:
                ix = max(a.x, b.x); iy = max(a.y, b.y)
                iw = min(a.x+a.w, b.x+b.w) - ix
                ih = min(a.y+a.h, b.y+b.h) - iy
                if iw>0 and ih>0: out.add_rect(ix, iy, iw, ih)
        return out
    def bounding_box(self):
        if not self.rects: return Rect(0,0,0,0)
        minx = min(r.x for r in self.rects); miny = min(r.y for r in self.rects)
        maxx = max(r.x+r.w for r in self.rects); maxy = max(r.y+r.h for r in self.rects)
        return Rect(minx, miny, maxx-minx, maxy-miny)

# ---------------- Paths ----------------
class Path:
    def __init__(self): self.cmds=[]
    def move_to(self,x,y): self.cmds.append(('M',float(x),float(y)))
    def line_to(self,x,y): self.cmds.append(('L',float(x),float(y)))
    def cubic_to(self,x1,y1,x2,y2,x3,y3): self.cmds.append(('C',float(x1),float(y1),float(x2),float(y2),float(x3),float(y3)))
    def close(self): self.cmds.append(('Z',))
    def flatten(self,tol=0.5):
        def cubic_flat(x0,y0,x1,y1,x2,y2,x3,y3,tol,out):
            ux,uy=x3-x0,y3-y0
            d1=abs((x1-x3)*uy-(y1-y3)*ux)
            d2=abs((x2-x3)*uy-(y2-y3)*ux)
            if d1+d2<tol: out.append((x3,y3)); return
            x01,y01=(x0+x1)/2,(y0+y1)/2
            x12,y12=(x1+x2)/2,(y1+y2)/2
            x23,y23=(x2+x3)/2,(y2+y3)/2
            x012,y012=(x01+x12)/2,(y01+y12)/2
            x123,y123=(x12+x23)/2,(y12+y23)/2
            x0123,y0123=(x012+x123)/2,(y012+y123)/2
            cubic_flat(x0,y0,x01,y01,x012,y012,x0123,y0123,tol,out)
            cubic_flat(x0123,y0123,x123,y123,x23,y23,x3,y3,tol,out)
        polys=[]; cur=[]; penx,peny=None,None
        for c in self.cmds:
            if c[0]=='M':
                if cur: polys.append(cur); cur=[]
                _,x,y=c; cur.append((x,y)); penx,peny=x,y
            elif c[0]=='L':
                _,x,y=c; cur.append((x,y)); penx,peny=x,y
            elif c[0]=='C':
                _,x1,y1,x2,y2,x3,y3=c; tmp=[]; cubic_flat(penx,peny,x1,y1,x2,y2,x3,y3,tol,tmp)
                if tmp: cur.extend(tmp); penx,peny=x3,y3
            elif c[0]=='Z':
                if cur: polys.append(cur); cur=[]
        if cur: polys.append(cur)
        return polys

# ---------------- DeviceContext ----------------
class DeviceContext:
    ROP_COPY='COPY'; ROP_XOR='XOR'; ROP_AND='AND'; ROP_OR='OR'
    def __init__(self, canvas:Canvas):
        self.canvas = canvas
        self.pen = Pen('*',1,'solid')
        self.brush = Brush('#',gradient=[' ','░','▒','▓','█'])
        self.font = Font()
        self.clip = Region([(0,0,canvas.w,canvas.h)])
        self.transform = (1.0,1.0,0.0,0.0,0.0) # scale_x, scale_y, trans_x, trans_y, rotation_degree
        self._undo = deque(maxlen=50)
        self._state_stack = deque(maxlen=50)        
    # ---------- Undo ----------
    def push_undo(self): self._undo.append(self.canvas.snapshot())
    def undo(self):
        if self._undo: self.canvas.restore_snapshot(self._undo.pop())
    
    # ---------- State stack ----------
    def save_state(self): self._state_stack.append(copy.deepcopy((self.pen,self.brush,self.font,self.clip,self.transform)))
    def restore_state(self):
        if self._state_stack:
            self.pen,self.brush,self.font,self.clip,self.transform=self._state_stack.pop()

    # ---------- Transform ----------
    def translate(self,tx,ty):
        sx,sy,ox,oy,r=self.transform
        self.transform=(sx,sy,ox+tx,oy+ty,r)
    def scale(self,sx,sy=None):
        if sy is None: sy=sx
        _,_,ox,oy,r=self.transform
        self.transform=(sx,sy,ox,oy,r)
    def rotate(self,deg):
        sx,sy,ox,oy,_=self.transform
        self.transform=(sx,sy,ox,oy,deg)
    def _apply_transform(self,x,y):
        sx,sy,tx,ty,rot=self.transform
        x = x*sx + tx; y = y*sy + ty
        rad = radians(rot)
        xr = x*cos(rad) - y*sin(rad)
        yr = x*sin(rad) + y*cos(rad)
        return int(round(xr)), int(round(yr))

    # ---------- Clip ----------
    def _in_clip(self,x,y): return self.clip.contains(x,y)
    def set_clip_rect(self,x,y,w,h): self.clip=Region([(x,y,w,h)])
    def set_clip_region(self,region): self.clip=region

    # ---------- Pixel ----------
    def set_pixel(self,x,y,ch=None,shade_val=None,alpha=1.0):
        xi,yi=self._apply_transform(x,y)
        if not self.canvas.in_bounds(xi,yi): return
        if not self._in_clip(xi,yi): return
        if shade_val is not None: ch=self.brush.get_char(xi,yi,val=shade_val)
        self.canvas.blend_pixel(xi,yi,ch,alpha)

    # ---------- Line AA ----------
    def line(self,x1,y1,x2,y2):
        self.push_undo()
        x1i,y1i=self._apply_transform(x1,y1)
        x2i,y2i=self._apply_transform(x2,y2)
        dx=x2i-x1i; dy=y2i-y1i
        steps=max(abs(dx),abs(dy))
        if steps==0: self.set_pixel(x1i,y1i); return
        for i in range(steps+1):
            t=i/steps
            x=x1i+dx*t; y=y1i+dy*t
            fx,fy=x-floor(x), y-floor(y)
            val=1-sqrt(fx**2+fy**2)
            self.set_pixel(floor(x),floor(y),shade_val=val)

    # ---------- Rect / fill ----------
    def rect(self,x,y,w,h,fill=False):
        self.push_undo()
        if fill:
            for yy in range(y,y+h):
                for xx in range(x,x+w):
                    val=(yy-y)/(h-1) if self.brush.gradient else None
                    self.set_pixel(xx,yy,shade_val=val)
        else:
            self.line(x,y,x+w-1,y)
            self.line(x,y,x,y+h-1)
            self.line(x+w-1,y,x+w-1,y+h-1)
            self.line(x,y+h-1,x+w-1,y+h-1)

    # ---------- Bitmap ----------
    def draw_bitmap(self,path,dx=0,dy=0,scale_w=1.0,scale_h=1.0,use_gradient=True):
        img=Image.open(path).convert('L')
        w,h=img.size
        w=int(w*scale_w); h=int(h*scale_h)
        img=img.resize((w,h))
        for y in range(h):
            for x in range(w):
                lum=img.getpixel((x,y))/255
                ch=self.brush.get_char(x,y,val=lum) if use_gradient else self.brush.ch
                self.set_pixel(dx+x,dy+y,ch)

    # ---------- Text ----------
    def text_out(self, x, y, text, align='left', rotate_deg=0, bold=False, underline=False, strike=False):
        self.push_undo()
        xi, yi = self._apply_transform(x, y)
        if align == 'center': xi -= len(text)//2
        elif align == 'right': xi -= len(text)
        rad = radians(rotate_deg)
        cosr, sinr = cos(rad), sin(rad)
        gradient = [' ', '░', '▒', '▓', '█']
        for i, ch in enumerate(text):
            tx = xi + i*cosr
            ty = yi + i*sinr
            fx, fy = tx - floor(tx), ty - floor(ty)
            shade_val = 1 - sqrt(fx**2 + fy**2)
            aa_ch = gradient[int(clamp(shade_val,0,1)*(len(gradient)-1))]
            self.set_pixel(int(round(tx)), int(round(ty)), aa_ch)
            if bold: self.set_pixel(int(round(tx))+1, int(round(ty)), aa_ch)
            if underline: self.set_pixel(int(round(tx)), int(round(ty))+1, '_')
            if strike: self.set_pixel(int(round(tx)), int(round(ty)), '-')

    # ---------- BitBlt ----------
    def bitblt(self, src_canvas, sx=0, sy=0, w=None, h=None, dx=0, dy=0, rop='COPY'):
        self.push_undo()
        w=w if w else src_canvas.w
        h=h if h else src_canvas.h
        for yy in range(h):
            for xx in range(w):
                sx2,sy2=sx+xx,sy+yy; dx2,dy2=dx+xx,dy+yy
                if 0<=sx2<src_canvas.w and 0<=sy2<src_canvas.h and 0<=dx2<self.canvas.w and 0<=dy2<self.canvas.h:
                    src = src_canvas.get(sx2,sy2)
                    dst = self.canvas.get(dx2,dy2)
                    if rop=='COPY': ch=src
                    elif rop=='XOR': ch=src if src!=dst else ' '
                    elif rop=='AND': ch=src if dst!=' ' else ' '
                    elif rop=='OR': ch=src if src!=' ' else dst
                    else: ch=src
                    self.canvas.set(dx2,dy2,ch)

    # ---------- Select objects ----------
    def select_pen(self, pen): self.pen=pen
    def select_brush(self, brush): self.brush=brush
    def select_font(self, font): self.font=font

    # ---------- Save ----------
    def save_to_file(self, path): self.canvas.save_text(path)

    # ---------- Bitmap 1-bit ----------
    def draw_bitmap_bw(self, path, dx=0, dy=0):
        img = Image.open(path).convert('1')
        w,h = img.size
        for y in range(h):
            for x in range(w):
                ch = '█' if img.getpixel((x,y)) else ' '
                self.set_pixel(dx+x, dy+y, ch)

    # ---------- ClearType subpixel ----------
    def set_pixel_cleartype(self, x, y, lum):
        subvals = [clamp(lum*1.0,0,1), clamp(lum*0.7,0,1), clamp(lum*0.4,0,1)]
        gradient = [' ', '░', '▒', '▓', '█']
        for i,v in enumerate(subvals):
            idx = int(v*(len(gradient)-1))
            self.set_pixel(x+i, y, gradient[idx])
class Layer:
    def __init__(self, w, h, alpha=1.0, visible=True, bg=' '):
        self.canvas = Canvas(w, h, bg)
        self.alpha = alpha
        self.visible = visible

class DeviceContextWithLayers(DeviceContext):
    def __init__(self, canvas:Canvas):
        super().__init__(canvas)
        self.layers = []
        self.background_layer = None

    # ---------- Background ----------
    def set_background_char(self, ch=' '):
        """Đặt background bằng ký tự."""
        if self.background_layer is None:
            self.background_layer = Layer(self.canvas.w, self.canvas.h)
        dc = DeviceContext(self.background_layer.canvas)
        dc.push_undo()
        dc.canvas.clear(ch)

    def set_background_bitmap(self, path):
        """Đặt background bằng bitmap."""
        if self.background_layer is None:
            self.background_layer = Layer(self.canvas.w, self.canvas.h)
        dc = DeviceContext(self.background_layer.canvas)
        dc.push_undo()
        dc.draw_bitmap(path, dx=0, dy=0, scale_w=1.0, scale_h=1.0, use_gradient=True)

    # ---------- Layer ----------
    def add_layer(self, alpha=1.0, visible=True, bg=' '):
        layer = Layer(self.canvas.w, self.canvas.h, alpha=alpha, visible=visible, bg=bg)
        self.layers.append(layer)
        return layer

    # ---------- Render all ----------
    def render_layers(self):
        # Render background
        if self.background_layer:
            for y in range(self.canvas.h):
                for x in range(self.canvas.w):
                    ch = self.background_layer.canvas.get(x,y)
                    self.canvas.set(x,y,ch)
        # Render each layer
        for layer in self.layers:
            if not layer.visible: continue
            for y in range(self.canvas.h):
                for x in range(self.canvas.w):
                    ch = layer.canvas.get(x,y)
                    self.canvas.blend_pixel(x, y, ch, alpha=layer.alpha)
def draw_bitmap_masked(self, path, dx=0, dy=0, scale_w=1.0, scale_h=1.0, use_gradient=True, mask_char=' '):
    img = Image.open(path).convert('L')
    w,h = img.size
    w=int(w*scale_w); h=int(h*scale_h)
    img=img.resize((w,h))
    for y in range(h):
        for x in range(w):
            lum = img.getpixel((x,y))/255
            ch = self.brush.get_char(x,y,val=lum) if use_gradient else self.brush.ch
            if ch != mask_char:   # chỉ vẽ nếu khác mask
                self.set_pixel(dx+x, dy+y, ch)
def text_out_spacing(self, x, y, text, spacing=0, kerning=None, **kwargs):
    """spacing: pixel giữa chữ, kerning: dict {('A','V'): -1}"""
    xi, yi = self._apply_transform(x, y)
    prev_ch = None
    for i, ch in enumerate(text):
        offset = 0
        if kerning and prev_ch:
            offset = kerning.get((prev_ch, ch), 0)
        self.set_pixel(xi + i*(1+spacing) + offset, yi, ch, **kwargs)
        prev_ch = ch                                    
def draw_path_aa(self, path: Path):
    """Vẽ Path (M,L,C,Z) với AA dựa trên flatten"""
    self.push_undo()
    polys = path.flatten(tol=0.5)  # flatten Bézier ra points
    for poly in polys:
        for i in range(len(poly)-1):
            x1,y1 = poly[i]
            x2,y2 = poly[i+1]
            self.line(x1, y1, x2, y2)  # dùng AA line đã có
def text_add(self, x, y, text, align='left', rotate_deg=0, bold=False, underline=False, strike=False,
                  spacing=0, kerning=None):
    """
    Text nâng cao dựa trên text_out gốc.
    - spacing: pixel giữa các ký tự
    - kerning: dict {('A','V'): offset_pixel} để điều chỉnh khoảng cách giữa ký tự
    - rotate_deg: xoay toàn bộ text
    """
    self.push_undo()  # lưu trạng thái trước khi vẽ

    # Tính vị trí từng ký tự với spacing & kerning
    positions = []
    cursor = 0
    prev_ch = None
    for ch in text:
        offset = kerning.get((prev_ch, ch), 0) if kerning and prev_ch else 0
        positions.append(cursor + offset)
        cursor += 1 + spacing + (offset or 0)
        prev_ch = ch

    # Canh lề
    total_width = positions[-1] + 1 if positions else 0
    if align == 'center':
        base_x = x - total_width // 2
    elif align == 'right':
        base_x = x - total_width
    else:
        base_x = x

    # Xoay toàn bộ text
    rad = radians(rotate_deg)
    cosr, sinr = cos(rad), sin(rad)

    for i, ch in enumerate(text):
        px = base_x + positions[i]
        py = y
        # transform xoay
        tx = px * cosr - py * sinr
        ty = px * sinr + py * cosr
        # gọi text_out gốc để tận dụng AA, brush, pen, clip, undo
        self.text_out(int(round(tx)), int(round(ty)), ch,
                      align='left', rotate_deg=0,
                      bold=bold, underline=underline, strike=strike)                                                                                    
DeviceContext.text_add = text_add
DeviceContext.draw_bitmap_masked = draw_bitmap_masked
DeviceContext.text_out_spacing = text_out_spacing
DeviceContext.draw_path_aa = draw_path_aa        
                                                                                
