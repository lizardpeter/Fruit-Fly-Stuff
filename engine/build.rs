use std::{fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders/scene.wgsl");
    let src = fs::read_to_string("shaders/scene.wgsl").expect("read shader");
    let module = naga::front::wgsl::parse_str(&src).expect("parse WGSL shader");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
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
}
