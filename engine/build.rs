use std::{fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders/scene.wgsl");
    let src = fs::read_to_string("shaders/scene.wgsl").expect("read shader");
    let module = naga::front::wgsl::parse_str(&src).expect("parse WGSL shader");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    ).validate(&module).expect("validate shader");
    let dest = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for (entry, stage, file) in [
        ("vs_main", naga::ShaderStage::Vertex, "scene.vert.spv"),
        ("fs_main", naga::ShaderStage::Fragment, "scene.frag.spv"),
    ] {
        let words = naga::back::spv::write_vec(
            &module, &info, &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                entry_point: entry.to_owned(), shader_stage: stage,
            }),
        ).expect("compile SPIR-V");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(dest.join(file), bytes).expect("save SPIR-V");
    }

    println!("cargo:rerun-if-changed=shaders/retina.wgsl");
    let src = fs::read_to_string("shaders/retina.wgsl").expect("read retina compute shader");
    let module = naga::front::wgsl::parse_str(&src).expect("parse retina compute WGSL");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty(),
    ).validate(&module).expect("validate retina shader");
    let words = naga::back::spv::write_vec(
        &module, &info, &naga::back::spv::Options::default(),
        Some(&naga::back::spv::PipelineOptions {
            entry_point: "sample_facets".to_owned(),
            shader_stage: naga::ShaderStage::Compute,
        }),
    ).expect("compile retina SPIR-V");
    fs::write(dest.join("retina.comp.spv"),
        words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())
        .expect("save retina SPIR-V");
}
