//! Vulkan compute sampler: 1442 real GPU optical averages per head-mounted
//! image. The results are read after the same render-submit fence, then fed
//! into the stateful receptor/photoadaptation module on CPU.
//!
//! Coordinates are provided by the CPU reference lattice; the compute shader
//! performs the nine-pixel optical averaging without reading full RGBA frames
//! into the CPU just for biological vision.

use ash::vk;
use std::{error::Error, ffi::CString};

use fruit_fly_engine::retina::{Facet, TOTAL_FACETS};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct GpuFacetSampler {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor: vk::DescriptorSet,
    coords: vk::Buffer,
    coords_memory: vk::DeviceMemory,
    output: vk::Buffer,
    output_memory: vk::DeviceMemory,
}

unsafe fn host_buffer(
    instance: &ash::Instance, device: &ash::Device, physical: vk::PhysicalDevice,
    bytes: vk::DeviceSize,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer = device.create_buffer(&vk::BufferCreateInfo::default()
        .size(bytes)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
    let req = device.get_buffer_memory_requirements(buffer);
    let properties = instance.get_physical_device_memory_properties(physical);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let index = (0..properties.memory_type_count).find(|&i|
        req.memory_type_bits & (1 << i) != 0
            && properties.memory_types[i as usize].property_flags.contains(flags)
    ).ok_or("No host-coherent memory for Vulkan retinal compute")?;
    let memory = device.allocate_memory(&vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(index), None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;
    Ok((buffer, memory))
}

impl GpuFacetSampler {
    pub unsafe fn new(
        instance: &ash::Instance, device: &ash::Device, physical: vk::PhysicalDevice,
        color_view: vk::ImageView, facets: &[Facet],
    ) -> Result<Self> {
        if facets.len() != TOTAL_FACETS { return Err("expected 1442 projected facets".into()); }
        let bytes = (TOTAL_FACETS * 16) as vk::DeviceSize;
        let (coords, coords_memory) = host_buffer(instance, device, physical, bytes)?;
        let (output, output_memory) = host_buffer(instance, device, physical, bytes)?;
        let mapped = device.map_memory(coords_memory, 0, bytes, vk::MemoryMapFlags::empty())? as *mut f32;
        for (i, facet) in facets.iter().enumerate() {
            let loc = mapped.add(i * 4);
            std::ptr::copy_nonoverlapping(
                [facet.pixel_x, facet.pixel_y, 0.0, 0.0].as_ptr(), loc, 4
            );
        }
        device.unmap_memory(coords_memory);

        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0)
                .descriptor_count(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1)
                .descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2)
                .descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None
        )?;
        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
        ];
        let pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1),
            None,
        )?;
        let sets = [layout];
        let descriptor = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sets)
        )?[0];
        let image = [vk::DescriptorImageInfo::default().image_view(color_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let coordinates = [vk::DescriptorBufferInfo::default().buffer(coords).offset(0).range(bytes)];
        let results = [vk::DescriptorBufferInfo::default().buffer(output).offset(0).range(bytes)];
        device.update_descriptor_sets(&[
            vk::WriteDescriptorSet::default().dst_set(descriptor).dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&image),
            vk::WriteDescriptorSet::default().dst_set(descriptor).dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&coordinates),
            vk::WriteDescriptorSet::default().dst_set(descriptor).dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&results),
        ], &[]);

        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None)?;
        let words: Vec<u32> = include_bytes!(concat!(env!("OUT_DIR"), "/retina.comp.spv"))
            .chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
        let entry = CString::new("sample_facets")?;
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry))
            .layout(pipeline_layout)];
        let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(),
            &pipeline_info, None).map_err(|(_, e)| e)?[0];
        device.destroy_shader_module(module, None);
        Ok(Self {
            layout, pool, pipeline_layout, pipeline, descriptor,
            coords, coords_memory, output, output_memory,
        })
    }

    /// The offscreen eye image must currently be TRANSFER_SRC_OPTIMAL, after
    /// Vulkan has copied it to a snapshot staging buffer.
    pub unsafe fn record(
        &self, device: &ash::Device, command: vk::CommandBuffer, eye_image: vk::Image,
    ) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let to_read = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(eye_image).subresource_range(range)];
        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(),
            &[], &[], &to_read);
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout, 0, &[self.descriptor], &[]);
        device.cmd_dispatch(command, TOTAL_FACETS.div_ceil(64) as u32, 1, 1);
        let to_host = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.output).offset(0).size(vk::WHOLE_SIZE)];
        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(),
            &[], &to_host, &[]);
    }

    /// Safe after the renderer waits for the submit fence; coherent output
    /// memory contains a vec4 of optical RGB per facet.
    pub unsafe fn read_rgb(&self, device: &ash::Device) -> Result<Vec<[f32; 3]>> {
        let bytes = (TOTAL_FACETS * 16) as vk::DeviceSize;
        let mapped = device.map_memory(self.output_memory, 0, bytes,
            vk::MemoryMapFlags::empty())? as *const f32;
        let mut samples = Vec::with_capacity(TOTAL_FACETS);
        for i in 0..TOTAL_FACETS {
            let values = std::slice::from_raw_parts(mapped.add(i * 4), 3);
            samples.push([values[0], values[1], values[2]]);
        }
        device.unmap_memory(self.output_memory);
        // A completely black image is valid sensory input (lights out).
        // Reject corrupt/nonfinite data, but never demand a bright pixel.
        if samples.iter().flatten().any(|x| !x.is_finite() || *x < -0.001 || *x > 1.001) {
            return Err("invalid Vulkan facet samples".into());
        }
        Ok(samples)
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.layout, None);
        device.destroy_buffer(self.coords, None);
        device.free_memory(self.coords_memory, None);
        device.destroy_buffer(self.output, None);
        device.free_memory(self.output_memory, None);
    }
}
