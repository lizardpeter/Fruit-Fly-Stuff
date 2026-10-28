//! Fly-eye interface contract. This module does NOT pretend to capture pixels yet.
//! The eventual Vulkan offscreen render target will fill VisionFrame.rgba8.

use glam::{Mat4, Vec3};

use crate::world::FlyState;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeCamera {
    pub width: u32,
    pub height: u32,
    pub horizontal_fov_degrees: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for EyeCamera {
    fn default() -> Self {
        Self {
            width: 128, height: 96, horizontal_fov_degrees: 150.0,
            near: 0.02, far: 40.0,
        }
    }
}

impl EyeCamera {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 || self.width > 8192 || self.height > 8192
            || !self.horizontal_fov_degrees.is_finite()
            || !(1.0..179.0).contains(&self.horizontal_fov_degrees)
            || !self.near.is_finite() || !self.far.is_finite()
            || self.near <= 0.0 || self.far <= self.near
        {
            return Err("invalid eye-camera parameters");
        }
        Ok(())
    }

    pub fn view_projection(&self, fly: &FlyState) -> Mat4 {
        let heading = fly.heading;
        let forward = Vec3::new(heading.sin(), 0.0, heading.cos());
        let eye = fly.position + Vec3::Y * 0.22 + forward * 0.3;
        let view = Mat4::look_at_rh(eye, eye + forward, Vec3::Y);
        // Turn horizontal FOV into vertical FOV based on the image aspect ratio.
        let hfov = self.horizontal_fov_degrees.to_radians();
        let vfov = 2.0 * ((hfov / 2.0).tan() * self.height as f32 / self.width as f32).atan();
        Mat4::perspective_rh(vfov, self.width as f32 / self.height as f32, self.near, self.far)
            * view
    }
}

pub struct VisionFrame {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl VisionFrame {
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, &'static str> {
        let expected = (width as usize).checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4)).ok_or("frame too large")?;
        if width == 0 || height == 0 || rgba8.len() != expected {
            return Err("RGBA frame byte count must equal width*height*4");
        }
        Ok(Self { width, height, rgba8 })
    }
}

/// Future bridge from an image (or compound-eye receptor samples) to neural
/// stimulation. Implementations must specify neuron IDs, encoding, and timing.
pub trait SensoryEncoder {
    fn encode(&mut self, frame: &VisionFrame) -> Vec<(u64, f32)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_is_finite_and_points_forward() {
        let cam = EyeCamera::default();
        cam.validate().unwrap();
        let pose = FlyState::default();
        let transformed = cam.view_projection(&pose).to_cols_array();
        assert!(transformed.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn frame_requires_exact_rgba_length() {
        assert!(VisionFrame::new(2, 3, vec![0; 24]).is_ok());
        assert!(VisionFrame::new(2, 3, vec![0; 23]).is_err());
    }
}
