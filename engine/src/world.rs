//! Minimal deterministic 3D fly world, decoupled from any neural controller.

use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct FlyState {
    pub position: Vec3,
    /// Yaw around world +Y; heading 0 faces +Z.
    pub heading: f32,
    pub wing_phase: f32,
}

impl Default for FlyState {
    fn default() -> Self {
        Self { position: Vec3::new(0.0, 0.65, 0.0), heading: 0.0, wing_phase: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MotorCommand {
    pub forward: f32,
    pub turn: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub transform: Mat4,
    pub color: [f32; 4],
    pub center: Vec3,
}

impl Object {
    pub fn cuboid(center: Vec3, size: Vec3, color: [f32; 4]) -> Self {
        Self { transform: Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, center), color, center }
    }

    pub fn oriented(center: Vec3, size: Vec3, rotation: Quat, color: [f32; 4]) -> Self {
        Self { transform: Mat4::from_scale_rotation_translation(size, rotation, center), color, center }
    }
}

#[derive(Default)]
pub struct World {
    pub fly: FlyState,
    pub time: f32,
    pub paused: bool,
    /// Opt-in deterministic bright-bar onset/offset experiment for renderer CI.
    pub visual_trial: bool,
}

impl World {
    /// Simple scripted locomotion prototype; NOT the output of a fly brain.
    pub fn step(&mut self, dt: f32, motor: MotorCommand) {
        if self.paused || !dt.is_finite() || dt <= 0.0 { return; }
        let dt = dt.min(0.05);
        self.fly.heading += motor.turn.clamp(-1.0, 1.0) * dt * 1.9;
        let forward = Vec3::new(self.fly.heading.sin(), 0.0, self.fly.heading.cos());
        self.fly.position += forward * motor.forward.clamp(-1.0, 1.0) * dt * 1.8;
        self.fly.position.x = self.fly.position.x.clamp(-8.0, 8.0);
        self.fly.position.z = self.fly.position.z.clamp(-8.0, 8.0);
        self.fly.wing_phase += dt * 45.0;
        self.time += dt;
    }

    /// Visual *proxy*, not rendered pixels or biological compound-eye input.
    /// Two landmark-bearing signals in fly-local coordinates, each in [0,1].
    /// Replace this with offscreen vision + a documented neuron mapping later.
    pub fn landmark_senses(&self) -> (f32, f32) {
        let forward = Vec3::new(self.fly.heading.sin(), 0.0, self.fly.heading.cos());
        let rightward = Vec3::new(self.fly.heading.cos(), 0.0, -self.fly.heading.sin());
        let mut left = 0.0_f32;
        let mut right = 0.0_f32;
        for target in [Vec3::new(2.2, 0.65, 2.0), Vec3::new(-2.2, 0.65, -1.6)] {
            let delta = target - self.fly.position;
            let distance = delta.length();
            if distance < 0.05 { continue; }
            let ray = delta / distance;
            let ahead = forward.dot(ray);
            if ahead <= 0.0 { continue; }
            let side = rightward.dot(ray);
            let strength = (ahead * side.abs() / (1.0 + 0.20 * distance)).clamp(0.0, 1.0);
            if side > 0.0 { right = right.max(strength); }
            else { left = left.max(strength); }
        }
        (left, right)
    }

    /// The first group of scene objects is the external environment;
    /// those are the ONLY objects visible to the fly's eye.
    pub fn objects_with_environment_count(&self) -> (Vec<Object>, usize) {
        let mut o = Vec::with_capacity(64);
        // The world is a broad green floor with a few unmistakable markers.
        o.push(Object::cuboid(Vec3::new(0.0, -0.06, 0.0), Vec3::new(18.0, 0.1, 18.0), [0.14, 0.33, 0.23, 1.0]));
        for x in -4..=4 {
            o.push(Object::cuboid(Vec3::new(x as f32 * 2.0, 0.005, 0.0), Vec3::new(0.025, 0.01, 18.0), [0.2, 0.42, 0.3, 1.0]));
            o.push(Object::cuboid(Vec3::new(0.0, 0.005, x as f32 * 2.0), Vec3::new(18.0, 0.01, 0.025), [0.2, 0.42, 0.3, 1.0]));
        }
        o.push(Object::cuboid(Vec3::new(2.2, 0.25, 2.0), Vec3::new(0.5, 0.5, 0.5), [0.84, 0.41, 0.21, 1.0]));
        o.push(Object::cuboid(Vec3::new(-2.2, 0.35, -1.6), Vec3::new(0.7, 0.7, 0.7), [0.31, 0.55, 0.86, 1.0]));
        // Deterministic visual event for actual GPU/retina validation:
        // high-contrast bright bar appears for 0.75 s then disappears.
        // Geometry belongs to the world and is visible to both cameras.
        if self.visual_trial && (0.75..1.50).contains(&self.time) {
            let forward = Vec3::new(self.fly.heading.sin(), 0.0, self.fly.heading.cos());
            o.push(Object::oriented(
                self.fly.position + forward * 3.0 + Vec3::Y * 0.22,
                Vec3::new(2.0, 1.8, 0.08),
                Quat::from_rotation_y(self.fly.heading),
                [1.0, 1.0, 1.0, 1.0],
            ));
        }
        let environment_count = o.len();

        let p = self.fly.position;
        let yaw = Quat::from_rotation_y(self.fly.heading);
        let mut part = |local: Vec3, scale: Vec3, rotation: Quat, color: [f32; 4]| {
            o.push(Object::oriented(p + yaw * local, scale, yaw * rotation, color));
        };
        // Stylized body/head/abdomen with independent eye and wing shapes.
        part(Vec3::ZERO, Vec3::new(0.36, 0.28, 0.48), Quat::IDENTITY, [0.25, 0.20, 0.12, 1.0]);
        part(Vec3::new(0.0, 0.0, -0.40), Vec3::new(0.32, 0.26, 0.57), Quat::IDENTITY, [0.64, 0.47, 0.26, 1.0]);
        part(Vec3::new(0.0, 0.08, 0.36), Vec3::splat(0.30), Quat::IDENTITY, [0.48, 0.31, 0.15, 1.0]);
        for side in [-1.0_f32, 1.0] {
            part(Vec3::new(side * 0.15, 0.12, 0.48), Vec3::new(0.12, 0.14, 0.15), Quat::IDENTITY, [0.76, 0.12, 0.12, 1.0]);
            part(Vec3::new(side * 0.31, 0.29, -0.03),
                 Vec3::new(0.48, 0.025, 0.23),
                 Quat::from_rotation_z(side * (0.35 + self.fly.wing_phase.sin() * 0.25)),
                 [0.74, 0.82, 0.91, 1.0]);
            for z in [-0.2_f32, 0.05, 0.25] {
                part(Vec3::new(side * 0.26, -0.25, z),
                     Vec3::new(0.025, 0.40, 0.025),
                     Quat::from_rotation_z(-side * 0.4),
                     [0.20, 0.15, 0.10, 1.0]);
            }
        }
        (o, environment_count)
    }

    pub fn objects(&self) -> Vec<Object> {
        self.objects_with_environment_count().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn locomotion_is_deterministic() {
        let mut a=World::default();
        let mut b=World::default();
        let input=MotorCommand{forward:0.8,turn:0.2};
        for _ in 0..25 {a.step(0.016,input);b.step(0.016,input);}
        assert_eq!(a.fly.position,b.fly.position);
        assert!(a.fly.position.z > 0.0);
    }
    #[test]
    fn paused_world_does_not_advance() {
        let mut w=World{paused:true,..World::default()};
        w.step(1.0,MotorCommand{forward:1.0,turn:0.0});
        assert_eq!(w.time,0.0);
    }
    #[test]
    fn left_and_right_sensors_follow_heading() {
        let mut w = World::default();
        let (_left, right) = w.landmark_senses();
        assert!(right > 0.0);
        w.fly.heading = std::f32::consts::FRAC_PI_2;
        let (left, _right) = w.landmark_senses();
        assert!(left > 0.0);
    }
    #[test]
    fn visual_trial_has_a_bright_object_only_in_the_stimulus_window() {
        let mut w = World::default();
        w.visual_trial = true;
        w.time = 0.5;
        let baseline = w.objects_with_environment_count().1;
        w.time = 1.0;
        let (objects, illuminated) = w.objects_with_environment_count();
        assert_eq!(illuminated, baseline + 1);
        assert_eq!(objects[illuminated - 1].color, [1.0; 4]);
        w.time = 1.6;
        assert_eq!(w.objects_with_environment_count().1, baseline);
    }
    #[test]
    fn scene_contains_body_and_landmarks() {
        let world=World::default();
        assert!(world.objects().len()>20);
    }
}
