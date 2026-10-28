//! Minimal custom Vulkan 1.3 renderer: no Bevy, Unity, or prebuilt engine.
//! Uses ash only for Vulkan FFI and winit only for native window events.
//!
//! This first milestone draws a small 3D world and an articulated fly-shaped
//! placeholder. Neural control and offscreen fly-eye pixels are NOT wired yet.

use std::{
    collections::HashSet,
    error::Error,
    ffi::CString,
    sync::Arc,
    time::{Duration, Instant},
};

use ash::{vk, Entry};
use glam::{Mat4, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use fruit_fly_engine::world::{MotorCommand, World};
mod retina_gpu;
mod eye;
use eye::EyePass;
use fruit_fly_engine::bridge::{Event as NeuralEvent, NeuralLink};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn shader_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

struct Renderer {
    // _entry holds the loader library open for the Vulkan function pointers.
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    queue: vk::Queue,
    swap_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    eye: Option<EyePass>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_ready: vk::Semaphore,
    render_done: vk::Semaphore,
    finished: vk::Fence,
}

impl Renderer {
    /// Vulkan object creation and ownership are deliberately isolated in here.
    fn new(window: &Window) -> Result<Self> {
        unsafe {
            let entry = Entry::load()?;
            let app = CString::new("Fruit-Fly-Stuff")?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app).application_version(1)
                .engine_name(&app).engine_version(1)
                .api_version(vk::API_VERSION_1_3);
            let display = window.display_handle()?.as_raw();
            let extensions = ash_window::enumerate_required_extensions(display)?;
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(extensions),
                None,
            )?;
            let surface = ash_window::create_surface(
                &entry, &instance, display, window.window_handle()?.as_raw(), None
            )?;
            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
            let selected = instance.enumerate_physical_devices()?.into_iter().find_map(|physical| {
                let props = instance.get_physical_device_properties(physical);
                if vk::api_version_major(props.api_version) < 1
                    || vk::api_version_minor(props.api_version) < 3 { return None; }
                instance.get_physical_device_queue_family_properties(physical)
                    .iter().enumerate().find_map(|(i, q)| {
                        let present = surface_loader.get_physical_device_surface_support(
                            physical, i as u32, surface
                        ).ok()?;
                        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && q.queue_flags.contains(vk::QueueFlags::COMPUTE) && present {
                            Some((physical, i as u32))
                        } else { None }
                    })
            }).ok_or("No Vulkan 1.3 graphics/present device. Update your GPU driver.")?;
            let (physical, family) = selected;
            let priorities = [1.0_f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family).queue_priorities(&priorities)];
            let enabled_extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default()
                .dynamic_rendering(true);
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&enabled_extensions)
                .push_next(&mut dynamic_rendering);
            let device = instance.create_device(physical, &device_info, None)?;
            let queue = device.get_device_queue(family, 0);
            let swap_loader = ash::khr::swapchain::Device::new(&instance, &device);

            let capabilities = surface_loader.get_physical_device_surface_capabilities(physical, surface)?;
            let formats = surface_loader.get_physical_device_surface_formats(physical, surface)?;
            if formats.is_empty() { return Err("Vulkan surface has no formats".into()); }
            let format = formats.iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
                .copied().unwrap_or(formats[0]);
            let requested = window.inner_size();
            let extent = if capabilities.current_extent.width == u32::MAX {
                vk::Extent2D {
                    width: requested.width.clamp(capabilities.min_image_extent.width,
                        capabilities.max_image_extent.width),
                    height: requested.height.clamp(capabilities.min_image_extent.height,
                        capabilities.max_image_extent.height),
                }
            } else { capabilities.current_extent };
            if extent.width == 0 || extent.height == 0 {
                return Err("Surface extent is zero".into());
            }
            let desired = capabilities.min_image_count + 1;
            let images_count = if capabilities.max_image_count == 0 { desired }
                else { desired.min(capabilities.max_image_count) };
            let alpha = [
                vk::CompositeAlphaFlagsKHR::OPAQUE,
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                vk::CompositeAlphaFlagsKHR::INHERIT,
            ].into_iter().find(|x| capabilities.supported_composite_alpha.contains(*x))
                .ok_or("No supported swapchain composite-alpha mode")?;
            let swap_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(images_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(alpha)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let swapchain = swap_loader.create_swapchain(&swap_info, None)?;
            let images = swap_loader.get_swapchain_images(swapchain)?;
            let image_views: Vec<_> = images.iter().map(|&image| {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format.format)
                        .subresource_range(vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1).layer_count(1)),
                    None
                )
            }).collect::<std::result::Result<Vec<_>, _>>()?;

            let depth_format = vk::Format::D32_SFLOAT;
            let depth_image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(depth_format)
                    .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                    .mip_levels(1).array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED), None)?;
            let needs = device.get_image_memory_requirements(depth_image);
            let mem = instance.get_physical_device_memory_properties(physical);
            let mem_index = (0..mem.memory_type_count)
                .find(|&i| (needs.memory_type_bits & (1 << i)) != 0
                    && mem.memory_types[i as usize]
                        .property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
                .ok_or("No device-local memory for depth attachment")?;
            let depth_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size).memory_type_index(mem_index), None)?;
            device.bind_image_memory(depth_image, depth_memory, 0)?;
            let depth_view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(depth_image).view_type(vk::ImageViewType::TYPE_2D)
                    .format(depth_format)
                    .subresource_range(vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1).layer_count(1)),
                None)?;

            // One persistently reusable CPU-visible storage buffer holds instance
            // transforms and colors. The vertex shader draws all boxes in one call.
            const MAX_OBJECTS: usize = 512;
            const OBJECT_STRIDE: usize = 80; // mat4x4 + vec4, std430
            let buffer_info = vk::BufferCreateInfo::default()
                .size((MAX_OBJECTS * OBJECT_STRIDE) as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let instance_buffer = device.create_buffer(&buffer_info, None)?;
            let buffer_needs = device.get_buffer_memory_requirements(instance_buffer);
            let host_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;
            let host_idx = (0..mem.memory_type_count)
                .find(|&i| (buffer_needs.memory_type_bits & (1 << i)) != 0
                    && mem.memory_types[i as usize].property_flags.contains(host_flags))
                .ok_or("No host-coherent memory for instance buffer")?;
            let instance_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(buffer_needs.size).memory_type_index(host_idx), None)?;
            device.bind_buffer_memory(instance_buffer, instance_memory, 0)?;
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::VERTEX)];
            let descriptor_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)?;
            let set_layouts = [descriptor_layout];
            let descriptor_set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&set_layouts))?[0];
            let stored = [vk::DescriptorBufferInfo::default()
                .buffer(instance_buffer).offset(0).range((MAX_OBJECTS * OBJECT_STRIDE) as u64)];
            let updates = [vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&stored)];
            device.update_descriptor_sets(&updates, &[]);

            let vertex_code = shader_words(include_bytes!(concat!(env!("OUT_DIR"), "/scene.vert.spv")));
            let fragment_code = shader_words(include_bytes!(concat!(env!("OUT_DIR"), "/scene.frag.spv")));
            let vert = device.create_shader_module(&vk::ShaderModuleCreateInfo::default()
                .code(&vertex_code), None)?;
            let frag = device.create_shader_module(&vk::ShaderModuleCreateInfo::default()
                .code(&fragment_code), None)?;
            let entry_name = CString::new("vs_main")?;
            let fragment_name = CString::new("fs_main")?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX).module(vert).name(&entry_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(&fragment_name),
            ];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)?;
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1).scissor_count(1);
            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE).line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true).depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            let color_attachment = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_attachment);
            let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dyns);
            let formats = [format.format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&formats)
                .depth_attachment_format(depth_format);
            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut rendering);
            let pipeline = device.create_graphics_pipelines(
                vk::PipelineCache::null(), &[pipeline_info], None
            ).map_err(|(_, err)| err)?[0];
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None)?;
            let command_buffer = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let image_ready = device.create_semaphore(&semaphore_info, None)?;
            let render_done = device.create_semaphore(&semaphore_info, None)?;
            let finished = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?;
            let eye = EyePass::new(&instance, &device, physical, format.format)?;
            Ok(Self {
                _entry: entry, instance, surface_loader, surface, device, queue,
                swap_loader, swapchain, images, image_views, extent,
                depth_image, depth_memory, depth_view, eye: Some(eye), layout, pipeline,
                instance_buffer, instance_memory, descriptor_layout, descriptor_pool, descriptor_set,
                command_pool, command_buffer, image_ready, render_done, finished,
            })
        }
    }

    fn visual_signals(&self) -> (f32, f32) {
        self.eye.as_ref().map_or((0.0, 0.0), EyePass::signals)
    }

    fn facet_signals(&self) -> &[f32] {
        self.eye.as_ref().map_or(&[], EyePass::facet_signals)
    }

    fn facet_off_signals(&self) -> &[f32] {
        self.eye.as_ref().map_or(&[], EyePass::facet_off_signals)
    }

    fn captured_eye_frames(&self) -> u64 {
        self.eye.as_ref().map_or(0, EyePass::captured_frames)
    }

    fn draw(&mut self, world: &World, orbit: f32, delta_seconds: f32) -> Result<()> {
        unsafe {
            self.device.wait_for_fences(&[self.finished], true, u64::MAX)?;
            if let Some(eye) = self.eye.as_mut() {
                eye.collect(&self.device, if world.paused { 0.0 } else { delta_seconds })?;
            }
            let (idx, _) = self.swap_loader.acquire_next_image(
                self.swapchain, u64::MAX, self.image_ready, vk::Fence::null()
            )?;
            self.device.reset_fences(&[self.finished])?;
            self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let cb = self.command_buffer;
            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            let depth_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1);
            let barriers = [
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.images[idx as usize]).subresource_range(color_range),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.depth_image).subresource_range(depth_range),
            ];
            self.device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(), &[], &[], &barriers);

            let sky = vk::ClearValue { color: vk::ClearColorValue {
                float32: [0.07, 0.12, 0.20, 1.0],
            }};
            let z_clear = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } };
            let color = [vk::RenderingAttachmentInfo::default()
                .image_view(self.image_views[idx as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(sky)];
            let depth = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(z_clear);
            let render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D::default().extent(self.extent))
                .layer_count(1)
                .color_attachments(&color)
                .depth_attachment(&depth);
            self.device.cmd_begin_rendering(cb, &render_info);
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_set_viewport(cb, 0, &[vk::Viewport {
                x: 0.0, y: 0.0, width: self.extent.width as f32,
                height: self.extent.height as f32, min_depth: 0.0, max_depth: 1.0,
            }]);
            self.device.cmd_set_scissor(cb, 0, &[vk::Rect2D::default().extent(self.extent)]);

            let target = world.fly.position;
            let eye = target + Vec3::new(orbit.sin() * 6.0, 3.6, orbit.cos() * 6.0);
            let view = Mat4::look_at_rh(eye, target, Vec3::Y);
            let aspect = self.extent.width as f32 / self.extent.height as f32;
            let projection = Mat4::perspective_rh(58.0_f32.to_radians(), aspect, 0.05, 100.0);
            let vp = projection * view;

            let (objects, environment_count) = world.objects_with_environment_count();
            if objects.len() > 256 { return Err("scene exceeds instance buffer capacity".into()); }
            // Fence was waited on at the beginning of the frame: the GPU
            // cannot still be reading this host-coherent buffer.
            let mapped = self.device.map_memory(self.instance_memory, 0,
                (512 * 80) as u64, vk::MemoryMapFlags::empty())? as *mut f32;
            let eye_vp = self.eye.as_ref().expect("eye camera initialized")
                .camera.view_projection(&world.fly);
            for (index, object) in objects.iter().enumerate() {
                for (view_index, vp_matrix) in [(0, vp), (1, eye_vp)] {
                    let matrix = (vp_matrix * object.transform).to_cols_array();
                    let mut data = [0.0_f32; 20];
                    data[..16].copy_from_slice(&matrix);
                    data[16..].copy_from_slice(&object.color);
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(), mapped.add((view_index * 256 + index) * 20), 20,
                    );
                }
            }
            self.device.unmap_memory(self.instance_memory);
            self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS,
                self.layout, 0, &[self.descriptor_set], &[]);
            self.device.cmd_draw(cb, 36, objects.len() as u32, 0, 0);
            self.device.cmd_end_rendering(cb);
            self.eye.as_mut().expect("eye initialized").record(
                &self.device, cb, self.pipeline, self.layout,
                self.descriptor_set, environment_count as u32,
            );
            let end = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.images[idx as usize]).subresource_range(color_range)];
            self.device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(), &[], &[], &end);
            self.device.end_command_buffer(cb)?;

            let wait = [self.image_ready];
            let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal = [self.render_done];
            let commands = [cb];
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(&wait).wait_dst_stage_mask(&stages)
                .command_buffers(&commands).signal_semaphores(&signal)];
            self.device.queue_submit(self.queue, &submit, self.finished)?;
            self.swap_loader.queue_present(self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal)
                    .swapchains(&[self.swapchain])
                    .image_indices(&[idx]))?;
            Ok(())
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.finished, None);
            self.device.destroy_semaphore(self.image_ready, None);
            self.device.destroy_semaphore(self.render_done, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_buffer(self.instance_buffer, None);
            self.device.free_memory(self.instance_memory, None);
            if let Some(eye) = self.eye.take() { eye.destroy(&self.device); }
            self.device.destroy_image_view(self.depth_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_memory, None);
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.swap_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: World,
    keys: HashSet<KeyCode>,
    camera_orbit: f32,
    autopilot: bool,
    frame_started: Instant,
    neural: Option<NeuralLink>,
    neural_ready: bool,
    neural_motor: MotorCommand,
    neural_age: f32,
    neural_clock: f32,
    rendered_frames: u32,
    smoke_frames: Option<u32>,
}
impl Default for App {
    fn default() -> Self {
        Self {
            window: None, renderer: None, world: World::default(),
            keys: HashSet::new(), camera_orbit: 0.6,
            autopilot: true, frame_started: Instant::now(),
            neural: NeuralLink::from_environment(),
            neural_ready: false,
            neural_motor: MotorCommand::default(),
            neural_age: 99.0,
            neural_clock: 0.0,
            rendered_frames: 0,
            smoke_frames: std::env::var("FRUIT_FLY_SMOKE_FRAMES")
                .ok().and_then(|n| n.parse::<u32>().ok()).filter(|&n| n > 0),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attributes = Window::default_attributes()
            .with_title("Fruit-Fly-Stuff | Vulkan 3D world (prototype)")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_resizable(false);
        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = Arc::new(window);
                match Renderer::new(&window) {
                    Ok(renderer) => {
                        println!("Vulkan 1.3 renderer ready. WASD: fly; arrows: camera; P: controller toggle; Space: pause; Esc: quit");
                        if self.neural.is_some() {
                            println!("Neural bridge configured; waiting for the Python worker.");
                        } else {
                            println!("No neural bridge configured; using scripted demo autopilot.");
                        }
                        self.renderer = Some(renderer);
                        self.window = Some(window);
                        self.frame_started = Instant::now();
                    }
                    Err(error) => {
                        eprintln!("Vulkan initialization failed: {error}");
                        event_loop.exit();
                    }
                }
            }
            Err(err) => { eprintln!("Window error: {err}"); event_loop.exit(); }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) { return; }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    if pressed { self.keys.insert(code); } else { self.keys.remove(&code); }
                    if pressed && !event.repeat {
                        match code {
                            KeyCode::Escape => event_loop.exit(),
                            KeyCode::Space => self.world.paused = !self.world.paused,
                            KeyCode::KeyP => {
                                self.autopilot = !self.autopilot;
                                println!("Controller: {}", if !self.autopilot { "keyboard" }
                                    else if self.neural.is_some() { "experimental neural" }
                                    else { "scripted autopilot" });
                            },
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = self.frame_started.elapsed().as_secs_f32().min(0.05);
                self.frame_started = Instant::now();
                if self.keys.contains(&KeyCode::ArrowLeft) { self.camera_orbit -= dt * 1.1; }
                if self.keys.contains(&KeyCode::ArrowRight) { self.camera_orbit += dt * 1.1; }
                let direction = (self.keys.contains(&KeyCode::KeyW) as i32
                    - self.keys.contains(&KeyCode::KeyS) as i32) as f32;
                let turn = (self.keys.contains(&KeyCode::KeyD) as i32
                    - self.keys.contains(&KeyCode::KeyA) as i32) as f32;
                self.neural_age += dt;
                if let Some(link) = self.neural.as_mut() {
                    while let Some(event) = link.poll() {
                        match event {
                            NeuralEvent::Ready { mode, neurons, edges } => {
                                self.neural_ready = true;
                                println!("Neural bridge ready: {mode} with {neurons} neurons, {edges} connections");
                            },
                            NeuralEvent::Motor { command, active, seq } => {
                                self.neural_motor = command;
                                self.neural_age = 0.0;
                                if seq % 60 == 0 {
                                    println!("Neural step {seq}: {active} firing cells, motor={command:?}");
                                }
                            },
                            NeuralEvent::Fault(message) => {
                                eprintln!("Neural bridge failed: {message}");
                                self.neural_ready = false;
                                self.neural_motor = MotorCommand::default();
                            },
                        }
                    }
                    // Fixed neural control cadence, separate from Vulkan rendering FPS.
                    if !self.world.paused {
                        // Never build up a backlog of stale sensory samples.
                        self.neural_clock = (self.neural_clock + dt).min(1.0 / 30.0);
                    }
                    if self.autopilot && !self.world.paused && self.neural_ready
                        && self.neural_clock >= 1.0 / 30.0 {
                        self.neural_clock -= 1.0 / 30.0;
                        // These features come from an independent fly-eye GPU
                        // render pass and actual copied pixels, not geometry.
                        let (left, right) = self.renderer.as_ref()
                            .map_or((0.0, 0.0), Renderer::visual_signals);
                        let facets = self.renderer.as_ref()
                            .map_or(&[][..], Renderer::facet_signals);
                        let off = self.renderer.as_ref()
                            .map_or(&[][..], Renderer::facet_off_signals);
                        link.send_senses(left, right, facets, off);
                    }
                }
                let motor = if self.autopilot {
                    if self.neural.is_some() {
                        if self.neural_ready && self.neural_age < 0.3 {
                            self.neural_motor
                        } else { MotorCommand::default() }
                    } else {
                        // Scripted movement only if no neural worker was requested.
                        MotorCommand {forward: 0.18, turn: (self.world.time * 0.28).sin() * 0.35}
                    }
                } else {
                    MotorCommand {forward: direction, turn}
                };
                self.world.step(dt, motor);
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Err(err) = renderer.draw(&self.world, self.camera_orbit, dt) {
                        eprintln!("Render failed (resizing/recreating swapchains is not implemented): {err}");
                        event_loop.exit();
                    } else {
                        self.rendered_frames += 1;
                        if self.rendered_frames % 60 == 0 {
                            let eye = renderer.visual_signals();
                            println!("Fly-eye GPU pixels captured={} left={:.3} right={:.3}",
                                renderer.captured_eye_frames(), eye.0, eye.1);
                        }
                        if self.smoke_frames.is_some_and(|max| self.rendered_frames >= max) {
                            println!("Vulkan smoke: presented {} frames", self.rendered_frames);
                            event_loop.exit();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Approximately 60 Hz redraw requests without a busy-spin.
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(16)
        ));
        if let Some(w) = self.window.as_ref() { w.request_redraw(); }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.renderer.take();
        self.window.take();
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
