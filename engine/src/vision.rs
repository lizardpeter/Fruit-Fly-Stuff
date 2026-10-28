//! Fly-eye perspective camera and GPU-readback visual feature extraction.
//! Still an approximation: not a biologically calibrated compound eye.

use glam::{Mat4, Vec3};
use std::{fs, io, path::Path};

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

    /// Save a 32-bit uncompressed BMP that Windows can open without plugins.
    /// BMP rows are bottom-up and channels are BGRA (our frame is RGBA).
    pub fn save_bmp(&self, path: &Path) -> io::Result<()> {
        use io::Write;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = u32::try_from(self.rgba8.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "image too large"))?;
        let file_size = 54_u32.checked_add(bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "BMP too large"))?;
        let mut out = fs::File::create(path)?;
        out.write_all(b"BM")?;
        out.write_all(&file_size.to_le_bytes())?;
        out.write_all(&[0u8; 4])?;
        out.write_all(&54u32.to_le_bytes())?;
        out.write_all(&40u32.to_le_bytes())?;
        out.write_all(&(self.width as i32).to_le_bytes())?;
        out.write_all(&(self.height as i32).to_le_bytes())?;
        out.write_all(&1u16.to_le_bytes())?;
        out.write_all(&32u16.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&bytes.to_le_bytes())?;
        out.write_all(&[0u8; 16])?;
        for row in (0..self.height as usize).rev() {
            for pixel in self.rgba8[row * self.width as usize * 4
                ..(row + 1) * self.width as usize * 4].chunks_exact(4) {
                out.write_all(&[pixel[2], pixel[1], pixel[0], pixel[3]])?;
            }
        }
        Ok(())
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
        // Require a strong chromatic contrast, so plain sky and floor
        // are not mistaken for blue/red landmarks.
        let orange = (r - g.max(b) - 0.16).max(0.0);
        let blue = (b - r.max(g) - 0.16).max(0.0);
        let salient = ((orange + blue) * 2.0).min(1.0);
        let side = if x < width / 2 { 0 } else { 1 };
        sum[side] += salient;
        count[side] += 1;
    }
    (
        (sum[0] / (count[0].max(1) as f32) * 40.0).clamp(0.0, 1.0),
        (sum[1] / (count[1].max(1) as f32) * 40.0).clamp(0.0, 1.0),
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
    fn clear_sky_and_green_floor_are_not_false_targets() {
        for rgba in [[75u8, 95, 125, 255], [20, 40, 65, 255],
                     [40, 90, 60, 255]] {
            let pixels = rgba.repeat(32);
            let frame = VisionFrame::new(8, 4, pixels).unwrap();
            let (left, right) = extract_landmark_signals(&frame);
            assert!(left == 0.0 && right == 0.0);
        }
    }
    #[test]
    fn bmp_output_is_windows_readable() {
        let path = std::env::temp_dir().join(format!("fruit-fly-eye-{}.bmp", std::process::id()));
        let frame = VisionFrame::new(2,1,vec![255,0,0,255, 0,0,255,255]).unwrap();
        frame.save_bmp(&path).unwrap();
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..2], b"BM");
        assert_eq!(data.len(), 62);
        assert_eq!(&data[54..58], &[0,0,255,255]);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn frame_requires_exact_rgba_length() {
        assert!(VisionFrame::new(2, 3, vec![0; 24]).is_ok());
        assert!(VisionFrame::new(2, 3, vec![0; 23]).is_err());
    }
}
