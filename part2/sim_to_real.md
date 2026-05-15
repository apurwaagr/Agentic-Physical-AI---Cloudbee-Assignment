# Part 2C — Sim-to-Real Transfer

## Two key mechanisms behind the sim-to-real gap

### 1. Contact / friction dynamics mismatch

Simulators (MuJoCo, PyBullet, Isaac Sim) approximate rigid-body contact using simplified analytical models with a small number of parameters (static friction coefficient μ_s, restitution, contact stiffness). Real surfaces exhibit **micro-geometry-dependent, velocity-varying friction** that no current simulator captures perfectly. When the robot grasps a real object, the contact forces are different from simulation: slippage occurs at predicted-stable grasps, or extra resistance prevents predicted-easy pushes. This is especially pronounced for irregular, textured, or deformable objects.

**Concrete mitigation — domain randomisation over contact parameters:**  
During simulation training, uniformly randomise the friction coefficient μ ∈ [0.1, 1.0] and contact stiffness across episodes (this is why `friction` is a key parameter in our `scene_generator.py`). Training over this distribution forces the policy to learn contact-robust behaviours rather than over-fitting to a single simulated friction value. At deployment, the real friction (wherever it falls in the training range) is implicitly handled.

---

### 2. Visual appearance domain gap

Simulated renderers produce images with perfect textures, uniform lighting, and no sensor noise. Real cameras produce images with lens distortion, motion blur, varying exposure, specular highlights on shiny objects, and background clutter that never appears in simulation. Vision-based policies (VLAs, imitation-learning policies with image input) exploit renderer artefacts as spurious features, causing near-zero real-world performance even when simulation performance is high.

**Concrete mitigation — photorealistic rendering + appearance randomisation:**  
Replace the default renderer with a physically-based one (e.g., BlenderProc, NVIDIA Isaac Replicator) and apply randomised textures, HDR environment maps (varied `light_int`), and random camera poses (`cam_yaw`). Additionally, apply **image augmentation** during policy training: colour jitter, Gaussian noise, random erasing. This destroys the policy's reliance on any single visual cue and generalises better to real sensor statistics.
