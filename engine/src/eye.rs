//! Second Vulkan camera rendered from the fly's head into an offscreen texture.
//! A staging buffer transfers its *real* pixels into a sensor encoder. This is
//! a perspective-camera approximation, not an insect compound-eye model.
use ash::vk;
use std::error::Error;
use fruit_fly_engine::vision::{EyeCamera, VisionFrame};
use fruit_fly_engine::retina::Retina;
use crate::retina_gpu::GpuFacetSampler;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const INSTANCES_PER_VIEW: u32 = 256;

unsafe fn make_image(
    instance: &ash::Instance, device: &ash::Device, physical: vk::PhysicalDevice,
    width: u32, height: u32, format: vk::Format,
    usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let image = device.create_image(&vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D).format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL).usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
    let req = device.get_image_memory_requirements(image);
    let properties = instance.get_physical_device_memory_properties(physical);
    let kind = (0..properties.memory_type_count).find(|&i|
        req.memory_type_bits & (1 << i) != 0 &&
        properties.memory_types[i as usize].property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    ).ok_or("No device-local eye image memory")?;
    let memory = device.allocate_memory(&vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(kind), None)?;
    device.bind_image_memory(image, memory, 0)?;
    let view = device.create_image_view(&vk::ImageViewCreateInfo::default()
        .image(image).view_type(vk::ImageViewType::TYPE_2D).format(format)
        .subresource_range(vk::ImageSubresourceRange::default()
            .aspect_mask(aspect).level_count(1).layer_count(1)), None)?;
    Ok((image, memory, view))
}

pub struct EyePass {
    pub camera: EyeCamera,
    color: vk::Image,
    color_memory: vk::DeviceMemory,
    color_view: vk::ImageView,
    depth: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    color_format: vk::Format,
    ready: bool,
    last_signals: (f32, f32),
    captured: u64,
    retina: Retina,
    gpu_sampler: GpuFacetSampler,
    facet_values: Vec<f32>,
}

impl EyePass {
    pub unsafe fn new(
        instance: &ash::Instance, device: &ash::Device,
        physical: vk::PhysicalDevice, swapchain_format: vk::Format,
    ) -> Result<Self> {
        let camera = EyeCamera::default();
        camera.validate()?;
        let (color, color_memory, color_view) = make_image(
            instance, device, physical, camera.width, camera.height, swapchain_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        let (depth, depth_memory, depth_view) = make_image(
            instance, device, physical, camera.width, camera.height, vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH,
        )?;
        let bytes = (camera.width as u64) * (camera.height as u64) * 4;
        let staging = device.create_buffer(&vk::BufferCreateInfo::default().size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
        let req = device.get_buffer_memory_requirements(staging);
        let properties = instance.get_physical_device_memory_properties(physical);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let kind = (0..properties.memory_type_count).find(|&i|
            req.memory_type_bits & (1 << i) != 0 &&
            properties.memory_types[i as usize].property_flags.contains(flags)
        ).ok_or("No host-visible coherent memory for fly-eye readback")?;
        let staging_memory = device.allocate_memory(&vk::MemoryAllocateInfo::default()
            .allocation_size(req.size).memory_type_index(kind), None)?;
        device.bind_buffer_memory(staging, staging_memory, 0)?;
        let retina = Retina::new(camera.width, camera.height)?;
        let gpu_sampler = GpuFacetSampler::new(instance, device, physical, color_view, &retina.facets)?;
        Ok(Self {
            camera, color, color_memory, color_view, depth, depth_memory,
            depth_view, staging, staging_memory, color_format: swapchain_format,
            ready: false, last_signals: (0.0, 0.0), captured: 0,
            retina, gpu_sampler,
            facet_values: vec![0.0; fruit_fly_engine::retina::TOTAL_FACETS],
        })
    }

    /// Call only after waiting on the previous frame's submission fence.
    pub unsafe fn collect(&mut self, device: &ash::Device) -> Result<()> {
        if !self.ready { return Ok(()); }
        let bytes = (self.camera.width as usize) * (self.camera.height as usize) * 4;
        let ptr = device.map_memory(self.staging_memory, 0, bytes as u64,
            vk::MemoryMapFlags::empty())? as *const u8;
        let pixels = std::slice::from_raw_parts(ptr, bytes).to_vec();
        device.unmap_memory(self.staging_memory);
        // Most swapchains are BGRA; the encoder explicitly handles format order.
        let is_bgra = matches!(self.color_format,
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB);
        let mut rgba = pixels;
        if is_bgra {
            for px in rgba.chunks_exact_mut(4) { px.swap(0, 2); }
        }
        let frame = VisionFrame::new(self.camera.width, self.camera.height, rgba)?;
        // Vulkan compute sampled the actual eye texture into 1442 optical
        // RGB readings. CPU performs the tiny stateful biophysical-style
        // receptor dynamics; the full RGB readback is only for snapshots.
        let rgb = self.gpu_sampler.read_rgb(device)?;
        let sample = self.retina.step_rgb_samples(&rgb, 1.0 / 60.0)?;
        self.last_signals = (
            sample.hemispheric_change[0],
            sample.hemispheric_change[1],
        );
        for (dest, graded) in self.facet_values.iter_mut().zip(sample.graded.iter()) {
            // Broadband photoreceptor response; R7 UV unavailable from RGB.
            // These magnitudes are uncalibrated. Send individual facets rather
            // than collapsing the retina to two binary inputs.
            *dest = (graded[0] * 5.0).clamp(0.0, 1.0);
        }
        // Optional snapshot of the fly's OWN view, not the human observer's.
        // Every ~120 frames; no disk I/O unless explicitly configured.
        if self.captured % 120 == 0 {
            if let Ok(path) = std::env::var("FRUIT_FLY_EYE_SNAPSHOT") {
                if let Err(error) = frame.save_bmp(std::path::Path::new(&path)) {
                    eprintln!("Fly-eye snapshot failed: {error}");
                }
            }
        }
        self.captured += 1;
        self.ready = false;
        Ok(())
    }

    pub fn signals(&self) -> (f32, f32) { self.last_signals }
    pub fn facet_signals(&self) -> &[f32] { &self.facet_values }
    pub fn captured_frames(&self) -> u64 { self.captured }

    pub unsafe fn record(
        &mut self, device: &ash::Device, cb: vk::CommandBuffer,
        pipeline: vk::Pipeline, layout: vk::PipelineLayout, descriptor: vk::DescriptorSet,
        objects: u32,
    ) {
        let cr = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let dr = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1);
        let barriers = [
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.color).subresource_range(cr),
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.depth).subresource_range(dr),
        ];
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(), &[], &[], &barriers);
        let sky = vk::ClearValue { color: vk::ClearColorValue { float32: [0.07, 0.12, 0.20, 1.0] }};
        let depth_clear = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0, stencil: 0
        }};
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(self.color_view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(sky)];
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_view).image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(depth_clear);
        let extent = vk::Extent2D { width: self.camera.width, height: self.camera.height };
        device.cmd_begin_rendering(cb, &vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(extent)).layer_count(1)
            .color_attachments(&color).depth_attachment(&depth));
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_set_viewport(cb, 0, &[vk::Viewport {
            x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        }]);
        device.cmd_set_scissor(cb, 0, &[vk::Rect2D::default().extent(extent)]);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS,
            layout, 0, &[descriptor], &[]);
        device.cmd_draw(cb, 36, objects, 0, INSTANCES_PER_VIEW);
        device.cmd_end_rendering(cb);
        let to_transfer = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.color).subresource_range(cr)];
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(),
            &[], &[], &to_transfer);
        let region = [vk::BufferImageCopy::default().buffer_offset(0)
            .buffer_row_length(0).buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: self.camera.width,
                height: self.camera.height, depth: 1 })];
        device.cmd_copy_image_to_buffer(cb, self.color, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.staging, &region);
        let buffer_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.staging).offset(0).size(vk::WHOLE_SIZE)];
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[],
            &buffer_barrier, &[]);
        self.gpu_sampler.record(device, cb, self.color);
        self.ready = true;
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        self.gpu_sampler.destroy(device);
        device.destroy_buffer(self.staging, None);
        device.free_memory(self.staging_memory, None);
        device.destroy_image_view(self.color_view, None);
        device.destroy_image(self.color, None);
        device.free_memory(self.color_memory, None);
        device.destroy_image_view(self.depth_view, None);
        device.destroy_image(self.depth, None);
        device.free_memory(self.depth_memory, None);
    }
}
