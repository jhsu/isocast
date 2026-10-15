//! Screens through the xdg-desktop-portal ScreenCast interface: the
//! compositor asks the user which display to share (once: the portal's
//! restore token is kept and reused), then hands over a PipeWire stream,
//! read with `pipewiresrc`.
//!
//! Compositors share DMA-BUFs in whatever layout their GPU uses (niri on
//! NVIDIA: XR24 in a block-linear tiling no CPU can read), so OpenGL
//! imports them (`glupload`, through EGL), converts to BGRA and copies
//! them out (`gldownload`); ordinary memory from other compositors takes
//! the same path. The engine's GPU then takes BGRA as it is, with no
//! Y'CbCr conversion on the CPU. Importing DMA-BUFs straight into wgpu
//! (zero-copy) can replace the round trip later.

use std::collections::BTreeMap;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use ashpd::desktop::PersistMode;
use ashpd::desktop::screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use isocast_engine::{CaptureFormat, VideoSink};

use crate::displays;
use crate::frames::{Frame, Layout};
use crate::pipeline::{Running, make};

/// How long the user has to answer the portal's dialog.
const PICK_TIMEOUT: Duration = Duration::from_secs(120);
/// How long the first frame may take once the stream is open.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);

/// Where the portal's restore tokens are kept, by display id.
fn tokens_path() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".isocast/linux-screencast.json"))
}

fn load_token(display: &str) -> Option<String> {
    let text = std::fs::read_to_string(tokens_path()?).ok()?;
    let tokens: BTreeMap<String, String> = serde_json::from_str(&text).ok()?;
    tokens.get(display).cloned()
}

/// Keep the token for next time. Tokens are single-use: every start
/// returns a new one.
fn save_token(display: &str, token: &str) {
    let Some(path) = tokens_path() else {
        return;
    };
    let mut tokens: BTreeMap<String, String> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    tokens.insert(display.into(), token.into());
    if let Some(folder) = path.parent() {
        let _ = std::fs::create_dir_all(folder);
    }
    if let Ok(text) = serde_json::to_string_pretty(&tokens) {
        let _ = std::fs::write(path, text);
    }
}

/// An open portal session: its PipeWire connection, and the thread that
/// keeps the session's D-Bus connection running until it's closed.
struct Portal {
    fd: OwnedFd,
    node: u32,
    close: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Portal {
    fn drop(&mut self) {
        let _ = self.close.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Ask the portal for a display's stream, showing its dialog unless a
/// restore token for `display` still holds.
fn open_portal(display: &str, cursor: bool) -> Result<Portal> {
    let token = load_token(display);
    let display = display.to_string();
    let (opened, wait) = mpsc::channel::<Result<(OwnedFd, u32)>>();
    let (close, closed) = mpsc::channel::<()>();
    let thread = thread::Builder::new()
        .name("isocast screencast portal".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    let _ = opened.send(Err(err.into()));
                    return;
                }
            };
            let session = runtime.block_on(async {
                let proxy = Screencast::new()
                    .await
                    .context("no screen-cast portal (xdg-desktop-portal)")?;
                let session = proxy.create_session(Default::default()).await?;
                proxy
                    .select_sources(
                        &session,
                        SelectSourcesOptions::default()
                            .set_cursor_mode(if cursor {
                                CursorMode::Embedded
                            } else {
                                CursorMode::Hidden
                            })
                            .set_sources(enumflags2::BitFlags::from(SourceType::Monitor))
                            .set_multiple(false)
                            .set_persist_mode(PersistMode::ExplicitlyRevoked)
                            .set_restore_token(token.as_deref()),
                    )
                    .await?
                    .response()?;
                let streams = proxy
                    .start(&session, None, Default::default())
                    .await?
                    .response()
                    .map_err(|err| anyhow!("screen sharing wasn't allowed: {err}"))?;
                if let Some(token) = streams.restore_token() {
                    save_token(&display, token);
                }
                let node = streams
                    .streams()
                    .first()
                    .context("the portal shared no display")?
                    .pipe_wire_node_id();
                let fd = proxy
                    .open_pipe_wire_remote(&session, Default::default())
                    .await?;
                anyhow::Ok((session, fd, node))
            });
            let session = match session {
                Ok((session, fd, node)) => {
                    let _ = opened.send(Ok((fd, node)));
                    session
                }
                Err(err) => {
                    let _ = opened.send(Err(err));
                    return;
                }
            };
            // Keep the session until the capture stops.
            let _ = closed.recv();
            let _ = runtime.block_on(session.close());
        })?;
    let (fd, node) = wait
        .recv_timeout(PICK_TIMEOUT)
        .context("no display was picked in the screen-sharing dialog in time")??;
    Ok(Portal {
        fd,
        node,
        close,
        thread: Some(thread),
    })
}

/// A frame's pixels as opaque BGRA, rows packed tightly, and its size.
/// A screen is opaque, but a compositor's AR24 may carry other alpha, and
/// the engine treats an RGB frame's alpha as real, so it's set.
fn bgra_pixels(buffer: &gst::BufferRef, caps: &gst::CapsRef) -> Result<(Vec<u8>, u32, u32)> {
    let info = gst_video::VideoInfo::from_caps(caps)?;
    let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info)
        .map_err(|_| anyhow!("could not read a screen frame"))?;
    let (width, height) = (info.width(), info.height());
    let row = width as usize * 4;
    let stride = frame.plane_stride()[0] as usize;
    let data = frame.plane_data(0)?;
    let mut pixels = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        pixels.extend_from_slice(&data[y * stride..y * stride + row]);
    }
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    Ok((pixels, width, height))
}

/// Stamp the stream's frames as they arrive and pass at most `fps` of
/// them on, before they're uploaded (where `element`'s sink pad is).
///
/// The stream's own timestamps don't advance (niri on PipeWire 1.6: every
/// buffer carries the first one's), so each frame gets the pipeline's
/// running time on arrival. Frames come when the screen changes, up to
/// the display's rate (144 Hz here); the first after each 1/`fps` tick
/// goes on, so a 144 Hz display gives a steady `fps`.
fn pace(element: &gst::Element, pipeline: &gst::Pipeline, fps: u32) {
    let period = gst::ClockTime::from_nseconds(1_000_000_000 / u64::from(fps.max(1)));
    let next = std::sync::Mutex::new(gst::ClockTime::ZERO);
    let pipeline = pipeline.downgrade();
    let pad = element
        .static_pad("sink")
        .expect("elements have a sink pad");
    pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
        let Some(pipeline) = pipeline.upgrade() else {
            return gst::PadProbeReturn::Ok;
        };
        let (Some(clock), Some(base)) = (pipeline.clock(), pipeline.base_time()) else {
            return gst::PadProbeReturn::Ok;
        };
        let now = clock.time().saturating_sub(base);
        let mut next = next.lock().unwrap_or_else(|e| e.into_inner());
        if now < *next {
            return gst::PadProbeReturn::Drop;
        }
        // The next tick, or a period from now after a gap.
        *next = (*next + period).max(now + period / 2);
        if let Some(gst::PadProbeData::Buffer(buffer)) = info.data.as_mut() {
            buffer.make_mut().set_pts(now);
        }
        gst::PadProbeReturn::Ok
    });
}

/// A running screen capture. Dropping it stops the stream, then closes
/// the portal session.
pub struct ScreenCapture {
    _running: Running,
    _portal: Portal,
}

/// Start capturing the display `query` names at up to `fps`, at its native
/// resolution.
pub fn capture_display(
    query: &str,
    fps: u32,
    cursor: bool,
    sink: VideoSink<Frame>,
) -> Result<(ScreenCapture, CaptureFormat)> {
    let display = displays::find(query)?;
    let portal = open_portal(&display.id, cursor)?;

    let pipeline = crate::pipeline::new()?;
    let source = make("pipewiresrc")?;
    source.set_property("fd", portal.fd.as_raw_fd());
    source.set_property("path", portal.node.to_string());
    let upload = make("glupload")?;
    let convert = make("glcolorconvert")?;
    let on_gpu = make("capsfilter")?;
    on_gpu.set_property(
        "caps",
        gst::Caps::builder("video/x-raw")
            .features(["memory:GLMemory"])
            .field("format", "BGRA")
            .build(),
    );
    let download = make("gldownload")?;
    let appsink = gst_app::AppSink::builder()
        .caps(
            &gst::Caps::builder("video/x-raw")
                .field("format", "BGRA")
                .build(),
        )
        .sync(false)
        .max_buffers(2)
        .drop(true)
        .build();
    let elements = [
        &source,
        &upload,
        &convert,
        &on_gpu,
        &download,
        appsink.upcast_ref(),
    ];
    pipeline.add_many(elements)?;
    gst::Element::link_many(elements).context("could not connect the screen stream")?;

    pace(&upload, &pipeline, fps);
    let (first, size) = mpsc::sync_channel::<(u32, u32)>(1);
    let frames = sink.frame;
    let pipeline_ref = pipeline.downgrade();
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let (Some(buffer), Some(caps), Some(pipeline)) =
                    (sample.buffer(), sample.caps(), pipeline_ref.upgrade())
                else {
                    return Ok(gst::FlowSuccess::Ok);
                };
                let Some(captured) = crate::pipeline::capture_time(&pipeline, buffer) else {
                    return Ok(gst::FlowSuccess::Ok);
                };
                let pixels = bgra_pixels(buffer, caps).map_err(|_| gst::FlowError::Error)?;
                let _ = first.try_send((pixels.1, pixels.2));
                let frame = Frame::from_pixels(pixels.1, pixels.2, Layout::Bgra, pixels.0)
                    .map_err(|_| gst::FlowError::Error)?;
                frames(frame, captured);
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    let running = Running::start(
        pipeline,
        format!("display {:?}", display.name),
        sink.stopped,
    )?;
    // The stream's size is only certain once a frame arrives (the portal
    // may report logical rather than pixel sizes).
    let (width, height) = size
        .recv_timeout(FIRST_FRAME_TIMEOUT)
        .with_context(|| format!("{:?} sent no picture", display.name))?;
    if width % 2 != 0 || height % 2 != 0 {
        bail!("{width}×{height} has an odd side, which the encoders can't take");
    }
    Ok((
        ScreenCapture {
            _running: running,
            _portal: portal,
        },
        CaptureFormat { width, height, fps },
    ))
}
