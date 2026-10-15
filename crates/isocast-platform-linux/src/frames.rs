//! Frames: pixels in memory, textures the GPU drew into, or both.
//!
//! The compositor draws into a frame's textures and the file writers read
//! its pixels, so a frame keeps both, with textures for each importer
//! (each GPU device) that has shown it, and a version saying which copies
//! hold its current picture. Reading a frame drawn on the GPU copies it
//! back (`gpu::download`); importing one filled in memory, or drawn by
//! another importer, uploads it. Zero-copy (DMA-BUF) can replace either
//! copy later without changing the engine.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Result, bail};

use crate::gpu::{self, Planes};

/// Frames from one pool that may be in use at once. If writers stall, the
/// source drops frames instead of using ever more memory.
const MAX_FRAMES_IN_FLIGHT: usize = 32;

/// How a frame's pixels are laid out in memory, rows packed tightly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 8-bit BGRA.
    Bgra,
    /// 8-bit 4:2:0 Y'CbCr, video range, Rec. 709: a luma plane, then a
    /// half-size plane of interleaved Cb and Cr.
    Nv12,
}

impl Layout {
    /// Bytes for a `width`×`height` frame.
    pub fn len(self, width: u32, height: u32) -> usize {
        let (width, height) = (width as usize, height as usize);
        match self {
            Self::Bgra => width * height * 4,
            Self::Nv12 => width * height + width * height.div_ceil(2),
        }
    }
}

/// A video frame. Cloning is cheap: clones share the same pixels.
#[derive(Clone)]
pub struct Frame(Arc<Shared>);

struct Shared {
    width: u32,
    height: u32,
    layout: Layout,
    state: Mutex<State>,
}

pub(crate) struct State {
    /// Bumped whenever the frame's picture changes (or is about to).
    pub version: u64,
    /// The pixels in memory; empty until first needed.
    pub cpu: Vec<u8>,
    /// The version `cpu` holds, if any.
    pub cpu_version: Option<u64>,
    /// Textures importers made for the frame, kept while it's recycled.
    pub planes: Vec<Planes>,
}

impl State {
    pub fn cpu_current(&self) -> bool {
        self.cpu_version == Some(self.version)
    }

    /// Textures holding the current picture, if any.
    pub fn current_planes(&self) -> Option<&Planes> {
        self.planes
            .iter()
            .find(|planes| planes.version == Some(self.version))
    }
}

impl Frame {
    fn new(width: u32, height: u32, layout: Layout) -> Self {
        Self(Arc::new(Shared {
            width,
            height,
            layout,
            state: Mutex::new(State {
                version: 0,
                cpu: Vec::new(),
                cpu_version: None,
                planes: Vec::new(),
            }),
        }))
    }

    /// A frame holding `pixels` (laid out as `layout` says, rows packed
    /// tightly), as captures deliver them.
    pub fn from_pixels(width: u32, height: u32, layout: Layout, pixels: Vec<u8>) -> Result<Self> {
        if pixels.len() != layout.len(width, height) {
            bail!(
                "a {width}×{height} {layout:?} frame needs {} bytes, not {}",
                layout.len(width, height),
                pixels.len()
            );
        }
        let frame = Self::new(width, height, layout);
        let mut state = frame.state();
        state.cpu = pixels;
        state.cpu_version = Some(state.version);
        drop(state);
        Ok(frame)
    }

    pub fn width(&self) -> u32 {
        self.0.width
    }

    pub fn height(&self) -> u32 {
        self.0.height
    }

    pub fn layout(&self) -> Layout {
        self.0.layout
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        // A panic while a frame was locked leaves its pixels as they were.
        self.0.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The frame's pixels, rows packed tightly, copied back from the GPU if
    /// it drew them.
    pub fn pixels(&self) -> Result<Vec<u8>> {
        let state = self.state();
        if state.cpu_current() {
            return Ok(state.cpu.clone());
        }
        match state.current_planes() {
            Some(planes) => gpu::download(planes, self.0.layout, self.0.width, self.0.height),
            None => bail!("the frame has no pixels yet"),
        }
    }

    /// Whether nothing but its pool holds the frame.
    fn unused(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

/// Frames of one size and layout, recycled once nothing downstream holds
/// them.
pub struct FramePool {
    width: u32,
    height: u32,
    layout: Layout,
    frames: Vec<Frame>,
}

impl FramePool {
    pub fn new(width: u32, height: u32, layout: Layout) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("a frame pool needs a size, not {width}×{height}");
        }
        Ok(Self {
            width,
            height,
            layout,
            frames: Vec::new(),
        })
    }

    fn next(&mut self) -> Option<Frame> {
        if let Some(frame) = self.frames.iter().find(|frame| frame.unused()) {
            return Some(frame.clone());
        }
        if self.frames.len() >= MAX_FRAMES_IN_FLIGHT {
            return None;
        }
        let frame = Frame::new(self.width, self.height, self.layout);
        self.frames.push(frame.clone());
        Some(frame)
    }
}

impl isocast_engine::FramePool for FramePool {
    type Frame = Frame;

    fn draw(&mut self, draw: &mut dyn FnMut(&mut [u8], usize)) -> Result<Option<Frame>> {
        if self.layout != Layout::Bgra {
            bail!("only BGRA frames are drawn on the CPU");
        }
        let Some(frame) = self.next() else {
            return Ok(None);
        };
        let mut state = frame.state();
        state
            .cpu
            .resize(self.layout.len(self.width, self.height), 0);
        state.version += 1;
        draw(&mut state.cpu, self.width as usize * 4);
        state.cpu_version = Some(state.version);
        drop(state);
        Ok(Some(frame))
    }

    fn take(&mut self) -> Result<Option<Frame>> {
        let Some(frame) = self.next() else {
            return Ok(None);
        };
        let mut state = frame.state();
        // The GPU is about to draw it: no copy holds its picture yet.
        state.version += 1;
        drop(state);
        Ok(Some(frame))
    }
}
