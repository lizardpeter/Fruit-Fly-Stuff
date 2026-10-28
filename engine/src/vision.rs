//! Fly-eye perspective camera and GPU-readback visual feature extraction.
//! Still an approximation: not a biologically calibrated compound eye.

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

/// Extract the salience of colored world landmarks in left/right halves of a
/// **rendered RGBA image**. This is an engineered color feature, not a retina.
/// Values are in [0,1]; the human observer's camera cannot influence them.
pub fn extract_landmark_signals(frame: &VisionFrame) -> (f32, f32) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut sum = [0.0_f32; 2];
    let mut count = [0_usize; 2];
    for (i, p) in frame.rgba8.chunks_exact(4).enumerate() {
        let x = i % width;
        let y = i / width;
        if y >= height { break; }
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        // Detect orange/red and blue objects, suppress green floor / blue sky.
        let orange = (r - g * 1.3).max(0.0) * (r - b * 1.1).max(0.0);
        let blue = (b - r * 1.25).max(0.0) * (b - g * 1.10).max(0.0);
        let salient = (orange + blue).min(1.0);
        let side = if x < width / 2 { 0 } else { 1 };
        sum[side] += salient;
        count[side] += 1;
    }
    (
        (sum[0] / (count[0].max(1) as f32) * 15.0).clamp(0.0, 1.0),
        (sum[1] / (count[1].max(1) as f32) * 15.0).clamp(0.0, 1.0),
    )
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
    fn rendered_color_features_are_hemisphere_specific() {
        let mut rgba = vec![0_u8; 8 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let i=(y*8+x)*4;
                rgba[i..i+4].copy_from_slice(&[230, 30, 20, 255]);
            }
        }
        let frame = VisionFrame::new(8,4,rgba).unwrap();
        let (left,right) = extract_landmark_signals(&frame);
        assert!(left>0.5 && right<0.01);
    }
    #[test]
    fn frame_requires_exact_rgba_length() {
        assert!(VisionFrame::new(2, 3, vec![0; 24]).is_ok());
        assert!(VisionFrame::new(2, 3, vec![0; 23]).is_err());
    }
}
