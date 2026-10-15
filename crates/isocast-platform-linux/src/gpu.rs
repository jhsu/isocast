//! Frames on the GPU: textures kept on each frame, uploads of frames
//! filled in memory, and copies back of frames the GPU drew.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use isocast_engine::gpu::{FrameImporter, Gpu, GpuFrame, GpuPixels, Matrix, YCbCr};

use crate::frames::{Frame, Layout, State};

/// The longest a copy back from the GPU may take.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(2);

/// Importers whose textures a frame keeps at once (the program, ISO
/// converters, monitors); the least recently used beyond this go.
const MAX_IMPORTERS_PER_FRAME: usize = 4;

/// How the engine's Y'CbCr frames encode colour.
const VIDEO_709: YCbCr = YCbCr {
    matrix: Matrix::Bt709,
    full_range: false,
    bits: 8,
};

/// A frame's textures for one importer: one for BGRA, luma and chroma for
/// Y'CbCr.
pub(crate) struct Planes {
    /// The importer that made them. Importers, not devices, own textures:
    /// devices from different wgpu instances can compare equal.
    importer: u64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    textures: Vec<wgpu::Texture>,
    /// The frame version the textures hold, if any.
    pub version: Option<u64>,
}

/// The texture format, size and bytes per texel of each plane.
fn plane_shapes(
    layout: Layout,
    width: u32,
    height: u32,
) -> Vec<(wgpu::TextureFormat, u32, u32, u32)> {
    match layout {
        Layout::Bgra => vec![(wgpu::TextureFormat::Bgra8Unorm, width, height, 4)],
        Layout::Nv12 => vec![
            (wgpu::TextureFormat::R8Unorm, width, height, 1),
            (
                wgpu::TextureFormat::Rg8Unorm,
                width.div_ceil(2),
                height.div_ceil(2),
                2,
            ),
        ],
    }
}

pub struct Importer {
    id: u64,
    gpu: Gpu,
}

impl Importer {
    pub fn new(gpu: &Gpu) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            gpu: gpu.clone(),
        }
    }

    /// Where in `state.planes` this importer's textures for `frame` are,
    /// made the first time (moved to the front: most recently used).
    fn planes(&self, frame: &Frame, state: &mut State) -> usize {
        if let Some(index) = state.planes.iter().position(|p| p.importer == self.id) {
            state.planes[..=index].rotate_right(1);
            return 0;
        }
        let textures = plane_shapes(frame.layout(), frame.width(), frame.height())
            .into_iter()
            .map(|(format, width, height, _)| {
                self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("frame"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            })
            .collect();
        state.planes.insert(
            0,
            Planes {
                importer: self.id,
                device: self.gpu.device.clone(),
                queue: self.gpu.queue.clone(),
                textures,
                version: None,
            },
        );
        // Drop the least recently used stale textures, never the new ones
        // or any holding the current picture.
        while state.planes.len() > MAX_IMPORTERS_PER_FRAME {
            let current = Some(state.version);
            let Some(stale) = state.planes[1..].iter().rposition(|p| p.version != current) else {
                break;
            };
            state.planes.remove(stale + 1);
        }
        0
    }

    /// The textures the GPU is about to draw `frame` into.
    fn draw_targets(&self, frame: &Frame, layout: Layout) -> Result<Vec<wgpu::Texture>> {
        if frame.layout() != layout {
            bail!(
                "a {:?} frame can't be drawn into as {layout:?}",
                frame.layout()
            );
        }
        let mut state = frame.state();
        let index = self.planes(frame, &mut state);
        let version = state.version;
        let planes = &mut state.planes[index];
        planes.version = Some(version);
        Ok(planes.textures.clone())
    }
}

impl FrameImporter for Importer {
    type Frame = Frame;

    fn import(&mut self, frame: &Frame) -> Result<GpuFrame> {
        let mut state = frame.state();
        let index = self.planes(frame, &mut state);
        let version = state.version;
        if state.planes[index].version != Some(version) {
            // The picture is in memory, or on another importer's textures
            // (copied back first).
            if !state.cpu_current() {
                let pixels = match state.current_planes() {
                    Some(planes) => {
                        download(planes, frame.layout(), frame.width(), frame.height())?
                    }
                    None => bail!("the frame has no pixels yet"),
                };
                state.cpu = pixels;
                state.cpu_version = Some(version);
            }
            let shapes = plane_shapes(frame.layout(), frame.width(), frame.height());
            let mut offset = 0;
            for (texture, (_, width, height, texel)) in
                state.planes[index].textures.iter().zip(shapes)
            {
                let len = (width * height * texel) as usize;
                self.gpu.queue.write_texture(
                    texture.as_image_copy(),
                    &state.cpu[offset..offset + len],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * texel),
                        rows_per_image: Some(height),
                    },
                    texture.size(),
                );
                offset += len;
            }
            state.planes[index].version = Some(version);
        }
        let textures = &state.planes[index].textures;
        let pixels = match frame.layout() {
            Layout::Bgra => GpuPixels::Rgb(textures[0].clone()),
            Layout::Nv12 => GpuPixels::YCbCr {
                luma: textures[0].clone(),
                chroma: textures[1].clone(),
                coding: VIDEO_709,
            },
        };
        Ok(GpuFrame {
            width: frame.width(),
            height: frame.height(),
            pixels,
        })
    }

    fn target(&mut self, frame: &Frame) -> Result<wgpu::Texture> {
        Ok(self.draw_targets(frame, Layout::Bgra)?.remove(0))
    }

    fn ycbcr_target(&mut self, frame: &Frame) -> Result<(wgpu::Texture, wgpu::Texture)> {
        let mut planes = self.draw_targets(frame, Layout::Nv12)?.into_iter();
        let luma = planes.next().expect("two planes");
        let chroma = planes.next().expect("two planes");
        Ok((luma, chroma))
    }

    // Every frame keeps its textures, so the cached forms are the same.
    fn import_cached(&mut self, frame: &Frame) -> Result<Option<GpuFrame>> {
        self.import(frame).map(Some)
    }

    fn ycbcr_target_cached(
        &mut self,
        frame: &Frame,
    ) -> Result<Option<(wgpu::Texture, wgpu::Texture)>> {
        self.ycbcr_target(frame).map(Some)
    }
}

/// Copy a frame's textures back into memory, rows packed tightly. The GPU
/// runs work in order, so this sees everything submitted before it.
pub(crate) fn download(
    planes: &Planes,
    layout: Layout,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let shapes = plane_shapes(layout, width, height);
    let mut encoder = planes
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame download"),
        });
    let mut buffers = Vec::new();
    for (texture, &(_, width, height, texel)) in planes.textures.iter().zip(&shapes) {
        // Copies into buffers need rows aligned to 256 bytes.
        let row = width * texel;
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = planes.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame download"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            texture.size(),
        );
        buffers.push((buffer, row, padded, height));
    }
    let submission = planes.queue.submit([encoder.finish()]);
    let (done, wait) = std::sync::mpsc::channel();
    for (buffer, ..) in &buffers {
        let done = done.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = done.send(result);
            });
    }
    planes
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(DOWNLOAD_TIMEOUT),
        })
        .context("the GPU did not copy a frame back in time")?;
    for _ in &buffers {
        wait.recv_timeout(DOWNLOAD_TIMEOUT)
            .context("a frame copy was never mapped")?
            .context("could not map a frame copy")?;
    }
    let mut pixels = Vec::with_capacity(layout.len(width, height));
    for (buffer, row, padded, height) in &buffers {
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .context("could not read a frame copy")?;
        for y in 0..*height as usize {
            let start = y * *padded as usize;
            pixels.extend_from_slice(&mapped[start..start + *row as usize]);
        }
    }
    Ok(pixels)
}
