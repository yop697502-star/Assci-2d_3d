# Ascii-2d_3d
# ASCII Graphics Engine (Python)

Dự án nhỏ thử nghiệm đồ họa bằng ký tự ASCII trong terminal.  
Mục tiêu chính là làm đồ họa 2D và 3D nhưng ascii,  
tự viết lại các bước transform, lighting, rasterization… bằng Python thuần.

Project chia làm 3 phần:

---

## 🚧 1. ASCII 3D Renderer

Một renderer 3D cơ bản chạy hoàn toàn trên CPU, gồm:

- Vec1 / Vec2 / Vec3 / Vec4  
- Ma trận 4×4 (translation, rotation, scale)  
- Camera LookAt + Perspective / Orthographic  
- Mesh: Cube, Sphere, OBJ loader đơn giản  
- Rasterizer tam giác (barycentric)  
- Depth buffer  
- Lambert & Phong lighting  
- ASCII shading nhiều mức sáng  
- Wireframe mode  
- Fog nhẹ  
- ShadowMap đơn giản  
- Scene graph cơ bản

Dùng để thử nghiệm pipeline 3D từ transform → projection → rasterize.

---

## 🧱 2. ASCII Dual Pipeline (Fixed-function & Core)

Bản mở rộng thử nghiệm hai pipeline khác nhau:

- Pipeline kiểu “OpenGL fixed-function” (ASCII 4-level)
- Pipeline kiểu “Core profile” dùng Renderer 3D chính

Phần này chủ yếu để học lại concept của OpenGL đời cũ và đời mới.

---

## ✏️ 3. ASCII GDI 2D Engine

Một engine 2D bằng ASCII, gồm:

- Canvas 2D  
- Pen / Brush / Font  
- Anti-aliased line  
- Bezier Path (flatten)  
- Bitmap loader (PIL)  
- Layer system + alpha blend  
- Clipping region  
- Text renderer đơn giản  
- Undo stack

Phần này dùng để thử mô phỏng “GDI-like API” nhưng bằng ký tự.

---

## ▶️ Chạy thử

Yêu cầu:
- Python 3.8+
- NumPy  
- Pillow (nếu dùng bitmap 2D)

pip install numpy pillow

Chạy demo tuỳ theo file bạn tổ chức:

python main.py

---

## 💬 Ghi chú

Dự án mang tính thử nghiệm,
Code có thể còn thay đổi hoặc chưa tối ưu.

