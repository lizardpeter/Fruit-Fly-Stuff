//! Experimental compound-eye transduction (not a calibrated retina).
//!
//! Two *projected* hexagonal facet mosaics, 721 facets per side.
//! Each facet samples the head-mounted Vulkan image with an optical
//! acceptance kernel, then eight dynamic receptor channels. The R7 UV
//! channel is explicitly unavailable from ordinary RGB: it is never faked.
//! Source FlyGym uses 721 bins/eye; this geometry is independently generated
//! and has NOT been aligned with the anatomical FAFB optic-lobe columns.

use crate::vision::VisionFrame;
use serde::{Deserialize, Serialize};

pub const FACETS_PER_EYE: usize = 721;
pub const TOTAL_FACETS: usize = FACETS_PER_EYE * 2;
pub const RECEPTORS_PER_FACET: usize = 8;
const HEX_RADIUS: i32 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side { Left, Right }

impl Side {
    pub fn code(self) -> u32 { if self == Self::Left { 0 } else { 1 } }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Facet {
    /// Stable per-eye facet index 0..721 in axial-hex order.
    pub index: u16,
    pub side: Side,
    pub hex_q: i16,
    pub hex_r: i16,
    /// Source image pixel center. A true calibration would be angular.
    pub pixel_x: f32,
    pub pixel_y: f32,
    /// Deterministic *illustrative* 30% pale / 70% yellow assignment.
    pub pale: bool,
}

#[derive(Clone, Debug)]
pub struct RetinaSample {
    /// 1442 facets × 8 receptor channel graded activations, in [0,1].
    /// R1–R6 are broad-band proxies, R7 is ZERO (no measured UV),
    /// and R8 is an RGB-derived chromatic proxy.
    pub graded: Vec<[f32; RECEPTORS_PER_FACET]>,
    /// Coarse left/right readout for legacy two-channel neural bridge;
    /// NOT the proper per-column connectome mapping.
    pub hemispheric_change: [f32; 2],
    pub uv_available: bool,
    pub facets: usize,
}

pub fn hex_facets(width: u32, height: u32) -> Result<Vec<Facet>, &'static str> {
    if width < 4 || height < 4 || width % 2 != 0 { return Err("retina expects even width >=4 and height >=4"); }
    let half = width as f32 * 0.5;
    let mut facets = Vec::with_capacity(TOTAL_FACETS);
    for side in [Side::Left, Side::Right] {
        let mut index = 0_u16;
        for q in -HEX_RADIUS..=HEX_RADIUS {
            for r in (-HEX_RADIUS).max(-q - HEX_RADIUS)..=(HEX_RADIUS).min(-q + HEX_RADIUS) {
                // A hexagonal lattice normalized into each half of a
                // **single** perspective view; this is not full 270° vision.
                let x = (3.0_f32.sqrt() * (q as f32 + r as f32 * 0.5))
                    / (3.0_f32.sqrt() * 3.0 * HEX_RADIUS as f32);
                let y = (1.5 * r as f32) / (3.0 * HEX_RADIUS as f32);
                // The lattice spans a circle-like patch with a margin.
                let local_x = (0.5 + x * 0.88) * (half - 1.0);
                let px = local_x + side.code() as f32 * half;
                let py = (0.5 - y * 0.88) * (height as f32 - 1.0);
                let pale = ((u32::from(index) * 73 + side.code() * 19) % 10) < 3;
                facets.push(Facet {
                    index, side, hex_q: q as i16, hex_r: r as i16,
                    pixel_x: px.clamp(side.code() as f32 * half, (side.code() + 1) as f32 * half - 1.0),
                    pixel_y: py.clamp(0.0, height as f32 - 1.0),
                    pale,
                });
                index += 1;
            }
        }
        if usize::from(index) != FACETS_PER_EYE { return Err("hex lattice did not produce 721 facets"); }
    }
    Ok(facets)
}

/// RGB *proxy*: UV never reconstructed from RGB. The broadband outer channels
/// and inner blue/green channels are not measured Drosophila opsin spectra.
pub fn spectral_proxy(rgb: [f32; 3], pale: bool) -> [f32; RECEPTORS_PER_FACET] {
    let [r, g, b] = rgb.map(|x| x.clamp(0.0, 1.0));
    let broadband = (0.21 * r + 0.72 * g + 0.07 * b).clamp(0.0, 1.0);
    let inner = if pale { (0.1 * g + 0.9 * b).min(1.0) }
        else { (0.9 * g + 0.1 * b).min(1.0) };
    [broadband, broadband, broadband, broadband, broadband, broadband, 0.0, inner]
}

fn sample_rgb(frame: &VisionFrame, facet: Facet) -> [f32; 3] {
    let mut sum = [0.0; 3];
    let mut weight_sum = 0.0;
    // Nine weighted pixels: approximate angular acceptance around one facet.
    for (dx, dy, w) in [
        (0,0,4.0),(1,0,2.0),(-1,0,2.0),(0,1,2.0),(0,-1,2.0),
        (1,1,1.0),(-1,1,1.0),(1,-1,1.0),(-1,-1,1.0),
    ] {
        let x = (facet.pixel_x.round() as i32 + dx)
            .clamp(facet.side.code() as i32 * (frame.width as i32 / 2),
                (facet.side.code() as i32 + 1) * (frame.width as i32 / 2) - 1) as usize;
        let y = (facet.pixel_y.round() as i32 + dy)
            .clamp(0, frame.height as i32 - 1) as usize;
        let ix = (y * frame.width as usize + x) * 4;
        for c in 0..3 { sum[c] += frame.rgba8[ix + c] as f32 / 255.0 * w; }
        weight_sum += w;
    }
    sum.map(|x| x / weight_sum)
}

pub struct Retina {
    pub facets: Vec<Facet>,
    fast: Vec<[f32; RECEPTORS_PER_FACET]>,
    adaptation: Vec<[f32; RECEPTORS_PER_FACET]>,
    initialized: bool,
    width: u32,
    height: u32,
}

impl Retina {
    pub fn new(width: u32, height: u32) -> Result<Self, &'static str> {
        let facets = hex_facets(width, height)?;
        Ok(Self {
            facets, fast: vec![[0.0; RECEPTORS_PER_FACET]; TOTAL_FACETS],
            adaptation: vec![[0.0; RECEPTORS_PER_FACET]; TOTAL_FACETS],
            initialized: false, width, height,
        })
    }

    /// State is continuous across frames. Both response and adaptation use
    /// illustrative time constants, not type-specific fly measurements.
    pub fn step(&mut self, image: &VisionFrame, dt_seconds: f32) -> Result<RetinaSample, &'static str> {
        if image.width != self.width || image.height != self.height
            || !dt_seconds.is_finite() || !(0.0..=1.0).contains(&dt_seconds) || dt_seconds == 0.0 {
            return Err("retina frame dimensions or timestep invalid");
        }
        let fast_alpha = 1.0 - (-dt_seconds / 0.025).exp();
        let slow_alpha = 1.0 - (-dt_seconds / 0.200).exp();
        let mut output = Vec::with_capacity(TOTAL_FACETS);
        let mut sum = [0.0; 2];
        for (i, facet) in self.facets.iter().copied().enumerate() {
            let input = spectral_proxy(sample_rgb(image, facet), facet.pale);
            let mut graded = [0.0; RECEPTORS_PER_FACET];
            for c in 0..RECEPTORS_PER_FACET {
                let prev = self.fast[i][c];
                let adapted = self.adaptation[i][c];
                self.fast[i][c] = prev + fast_alpha * (input[c] - prev);
                self.adaptation[i][c] += slow_alpha * (input[c] - adapted);
                // Rectified temporal contrast after slow light adaptation.
                graded[c] = (self.fast[i][c] - self.adaptation[i][c]).max(0.0).min(1.0);
            }
            // Aggregate broadband R1–6 change for legacy bridge.
            sum[facet.side.code() as usize] += graded[0];
            output.push(graded);
        }
        self.initialized = true;
        Ok(RetinaSample {
            graded: output,
            hemispheric_change: [
                (sum[0] / FACETS_PER_EYE as f32 * 5.0).min(1.0),
                (sum[1] / FACETS_PER_EYE as f32 * 5.0).min(1.0),
            ],
            uv_available: false,
            facets: TOTAL_FACETS,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(left: [u8; 4], right: [u8; 4]) -> VisionFrame {
        let mut data = Vec::new();
        for _ in 0..96 {
            for _ in 0..64 { data.extend(left); }
            for _ in 0..64 { data.extend(right); }
        }
        VisionFrame::new(128, 96, data).unwrap()
    }
    #[test]
    fn lattice_has_exact_counts_stable_indices_and_separate_eyes() {
        let faces = hex_facets(128, 96).unwrap();
        assert_eq!(faces.len(), 1442);
        assert_eq!(faces.iter().filter(|f| f.side == Side::Left).count(), 721);
        assert_eq!(faces[720].index, 720);
        assert_eq!(faces[721].index, 0);
        assert!(faces.iter().take(721).all(|f| f.pixel_x < 64.0));
        assert!(faces.iter().skip(721).all(|f| f.pixel_x >= 64.0));
        assert!(faces.iter().all(|f| f.pixel_y >= 0.0 && f.pixel_y < 96.0));
    }
    #[test]
    fn r7_uv_is_unavailable_not_invented() {
        let a = spectral_proxy([0.9, 0.3, 0.1], false);
        assert_eq!(a[6], 0.0);
        let b = spectral_proxy([0.1, 0.2, 0.9], true);
        assert!(b[7] > a[7]);
    }
    #[test]
    fn temporal_signals_have_eye_laterality_and_adapt() {
        let mut retina = Retina::new(128, 96).unwrap();
        let frame = image([255, 255, 255, 255], [0, 0, 0, 255]);
        let first = retina.step(&frame, 0.016).unwrap();
        assert_eq!(first.facets, 1442);
        assert!(!first.uv_available);
        assert!(first.hemispheric_change[0] > 0.0);
        assert!(first.hemispheric_change[1] < 0.0001);
        let mut last = first.hemispheric_change[0];
        for _ in 0..100 { last = retina.step(&frame, 0.016).unwrap().hemispheric_change[0]; }
        assert!(last < first.hemispheric_change[0] * 0.15);
    }
    #[test]
    fn rejects_invalid_resolution_and_time() {
        assert!(Retina::new(127, 96).is_err());
        let mut retina = Retina::new(128, 96).unwrap();
        assert!(retina.step(&image([0;4],[0;4]), -1.0).is_err());
    }
}
