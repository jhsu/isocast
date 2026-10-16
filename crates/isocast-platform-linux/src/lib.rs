//! Linux back end for Isocast.
//!
//! The master clock is `CLOCK_MONOTONIC`, which GStreamer and PipeWire
//! stamp buffers on. Frames are pixels in memory or textures the GPU drew
//! into (see [`frames`]); files are written by GStreamer, which encodes
//! with NVENC or VA-API where the machine has them.
//!
//! So far it records synthetic sources (PLAN.md L.1). Capture from
//! devices, media files and streaming come next; until then they fail
//! with an error naming what's missing.
#![cfg(target_os = "linux")]

use std::path::Path;
use std::sync::Arc;

use anyhow::bail;
use isocast_engine::gpu::Gpu;
use isocast_engine::{
    AudioRender, AudioSink, BrowserControl, BrowserPage, CameraRequest, CaptureFormat,
    EncodedVideo, JoinSpec, Joined, LiveAudioSpec, LiveVideoSpec, MediaInfo, MediaPart,
    OnJoinProgress, Platform, Probe, ScreenTarget, Time, VideoSink, Volume, WriterSpec,
};
use isocast_protocol::api::Machine;
use isocast_protocol::devices::{
    Devices, Permission, PermissionAction, PermissionDetail, PermissionKind, Permissions,
    PermissionsReport,
};
use isocast_protocol::{Canvas, SourceKind};

mod audio;
mod camera;
mod devices;
mod displays;
pub mod frames;
mod gpu;
mod pipeline;
mod screen;
mod system;
mod writer;

pub use frames::{Frame, FramePool, Layout};
pub use gpu::Importer;
pub use pipeline::Running;
pub use system::host_time;
pub use writer::Writer;

#[derive(Clone, Copy)]
pub struct LinuxPlatform;

/// Something this back end can't do yet, and the PLAN.md task that adds it.
fn not_yet<T>(what: &str, task: &str) -> anyhow::Result<T> {
    bail!("{what} isn't supported on Linux yet (PLAN.md {task})")
}

/// A running capture of any kind. Dropping it stops the capture.
pub enum Capture {
    Audio(pipeline::Running),
    Camera(pipeline::Running),
    Screen(screen::ScreenCapture),
}

/// Audio playing on an output device. Dropping it stops playback.
pub type Playback = pipeline::Running;

/// Uninhabited stand-ins for what this back end can't make yet: no value
/// of them exists, so their trait methods can never run.
pub enum MediaReader {}
pub enum VideoEncoder {}
pub enum AudioEncoder {}

impl isocast_engine::MediaReader for MediaReader {
    type Frame = Frame;

    fn next_video(&mut self) -> anyhow::Result<Option<(Frame, std::time::Duration)>> {
        match *self {}
    }

    fn next_audio(&mut self) -> anyhow::Result<Option<(Vec<f32>, std::time::Duration)>> {
        match *self {}
    }
}

impl isocast_engine::VideoEncoder for VideoEncoder {
    type Frame = Frame;

    fn encode(&mut self, _frame: &Frame, _pts: Time, _keyframe: bool) -> anyhow::Result<()> {
        match *self {}
    }

    fn receive(&mut self) -> anyhow::Result<Option<EncodedVideo>> {
        match *self {}
    }

    fn receive_timeout(
        &mut self,
        _timeout: std::time::Duration,
    ) -> anyhow::Result<Option<EncodedVideo>> {
        match *self {}
    }

    fn dropped(&self) -> u64 {
        match *self {}
    }

    fn set_rate(&mut self, _fps: u32, _bitrate_kbps: u32) -> anyhow::Result<()> {
        match *self {}
    }
}

impl isocast_engine::AudioEncoder for AudioEncoder {
    fn encode(&mut self, _samples: &[f32]) -> anyhow::Result<Vec<Vec<u8>>> {
        match *self {}
    }

    fn delay(&self) -> u32 {
        match *self {}
    }
}

/// Linux has no per-app capture permissions: devices are open to the
/// user's processes, and screens are chosen in the portal's dialog when
/// capture starts.
fn permissions_report() -> PermissionsReport {
    let details = PermissionKind::ALL
        .into_iter()
        .map(|permission| PermissionDetail {
            permission,
            state: Permission::Granted,
            needed_for: match permission {
                PermissionKind::ScreenRecording => "displays, windows and system audio",
                PermissionKind::Camera => "cameras",
                PermissionKind::Microphone => "microphones",
                PermissionKind::SystemAudio => "system audio",
            }
            .into(),
            action: PermissionAction::None,
            settings_url: String::new(),
            next_step: None,
            restart_after_granting: false,
            app: "isocastd".into(),
            app_path: None,
            denied_without_prompt: false,
        })
        .collect();
    PermissionsReport {
        permissions: Permissions {
            screen_recording: Permission::Granted,
            camera: Permission::Granted,
            microphone: Permission::Granted,
            system_audio: Permission::Granted,
        },
        app: "isocastd".into(),
        details,
    }
}

impl Platform for LinuxPlatform {
    type Frame = Frame;
    type FramePool = FramePool;
    type Writer = Writer;
    type Capture = Capture;
    type Importer = Importer;
    type Media = MediaReader;
    type Playback = Playback;
    type VideoEncoder = VideoEncoder;
    type AudioEncoder = AudioEncoder;
    type SharedSurface = isocast_surfaces::Surface;

    fn now(&self) -> Time {
        host_time()
    }

    fn devices(&self) -> anyhow::Result<Devices> {
        let mut notes =
            vec!["windows can't be listed on Linux: the portal picks them (PLAN.md L.6)".into()];
        let displays = displays::displays().unwrap_or_else(|err| {
            notes.push(format!("{err:#}"));
            Vec::new()
        });
        Ok(Devices {
            displays,
            cameras: camera::cameras()?,
            microphones: devices::microphones()?,
            audio_outputs: devices::audio_outputs()?,
            permissions: permissions_report().permissions,
            notes,
            ..Devices::default()
        })
    }

    fn permissions(&self) -> anyhow::Result<PermissionsReport> {
        Ok(permissions_report())
    }

    fn request_permissions(
        &self,
        _only: Option<PermissionKind>,
    ) -> anyhow::Result<PermissionsReport> {
        Ok(permissions_report())
    }

    fn open_permission_settings(&self, _permission: PermissionKind) -> anyhow::Result<String> {
        bail!("Linux has no permission settings to open: capture is open to your user")
    }

    fn media_info(&self, _path: &Path) -> anyhow::Result<MediaInfo> {
        not_yet("reading media files", "L.6")
    }

    fn open_media(&self, _path: &Path, _part: MediaPart) -> anyhow::Result<MediaReader> {
        not_yet("reading media files", "L.6")
    }

    fn frame_importer(&self, gpu: &Gpu) -> anyhow::Result<Importer> {
        Ok(Importer::new(gpu))
    }

    fn machine(&self) -> Machine {
        system::machine()
    }

    fn probe(&self, source: &SourceKind, canvas: &Canvas) -> anyhow::Result<Option<Probe>> {
        match source {
            SourceKind::Display(settings) => {
                let display = displays::find(&settings.display)?;
                Ok(Some(Probe {
                    device: display.name,
                    video: Some(CaptureFormat {
                        width: display.width,
                        height: display.height,
                        fps: canvas.fps,
                    }),
                    channels: None,
                }))
            }
            SourceKind::Window(_) | SourceKind::Application(_) => {
                not_yet("window and app capture", "L.6")
            }
            SourceKind::Camera(settings) => {
                let request = CameraRequest {
                    size: settings.width.zip(settings.height),
                    fps: settings.fps,
                };
                let (device, format) = camera::probe(&settings.device, request, canvas)?;
                Ok(Some(Probe {
                    device,
                    video: Some(format),
                    channels: None,
                }))
            }
            SourceKind::Microphone(settings) => {
                let found = devices::find("Audio/Source", "microphone", &settings.device)?;
                Ok(Some(Probe {
                    channels: Some(devices::channels(&found.device)),
                    device: found.name,
                    video: None,
                }))
            }
            SourceKind::SystemAudio(_) => Ok(Some(Probe {
                device: "system audio".into(),
                video: None,
                channels: Some(2),
            })),
            SourceKind::ApplicationAudio(_) => not_yet("application audio capture", "L.6"),
            _ => Ok(None),
        }
    }

    fn volume(&self, path: &Path) -> anyhow::Result<Volume> {
        system::volume(path)
    }

    fn write_latency(&self, probe: &Path) -> Option<std::time::Duration> {
        system::write_latency(probe).ok()
    }

    fn frame_pool(&self, width: u32, height: u32) -> anyhow::Result<FramePool> {
        FramePool::new(width, height, Layout::Bgra)
    }

    fn ycbcr_frame_pool(&self, width: u32, height: u32) -> anyhow::Result<FramePool> {
        FramePool::new(width, height, Layout::Nv12)
    }

    fn thread_cpu_time(&self) -> Option<std::time::Duration> {
        system::thread_cpu_time()
    }

    fn frame_size(&self, frame: &Frame) -> (u32, u32) {
        (frame.width(), frame.height())
    }

    fn frame_is_ycbcr(&self, frame: &Frame) -> bool {
        frame.layout() == Layout::Nv12
    }

    fn shared_surface(
        &self,
        _gpu: &Gpu,
        _width: u32,
        _height: u32,
    ) -> anyhow::Result<(isocast_surfaces::Surface, wgpu::Texture)> {
        not_yet("live monitors in other apps", "L.6")
    }

    fn encode_jpeg(
        &self,
        _frame: &Frame,
        _max_width: Option<u32>,
        _quality: f32,
    ) -> anyhow::Result<(Vec<u8>, u32, u32)> {
        not_yet("JPEG monitors", "L.6")
    }

    fn create_writer(&self, spec: &WriterSpec) -> anyhow::Result<Writer> {
        Writer::create(spec)
    }

    fn capture_screen(
        &self,
        target: ScreenTarget<'_>,
        fps: u32,
        _ten_bit: bool,
        cursor: bool,
        sink: VideoSink<Frame>,
    ) -> anyhow::Result<(Capture, CaptureFormat)> {
        let ScreenTarget::Display(display) = target else {
            return not_yet("window and app capture", "L.6");
        };
        let (capture, format) = screen::capture_display(display, fps, cursor, sink)?;
        Ok((Capture::Screen(capture), format))
    }

    fn capture_camera(
        &self,
        device: &str,
        request: CameraRequest,
        canvas: &Canvas,
        sink: VideoSink<Frame>,
    ) -> anyhow::Result<(Capture, CaptureFormat)> {
        let (capture, format) = camera::capture_camera(device, request, canvas, sink)?;
        Ok((Capture::Camera(capture), format))
    }

    fn capture_microphone(&self, device: &str, sink: AudioSink) -> anyhow::Result<(Capture, u32)> {
        let (capture, channels) = audio::capture_microphone(device, sink)?;
        Ok((Capture::Audio(capture), channels))
    }

    fn capture_system_audio(
        &self,
        exclude: &[String],
        sink: AudioSink,
        _exclude_own: bool,
    ) -> anyhow::Result<(Capture, u32)> {
        if !exclude.is_empty() {
            bail!(
                "system_audio.exclude isn't supported on Linux yet: remove {exclude:?} \
                 to record everything the computer plays"
            );
        }
        let (capture, channels) = audio::capture_system_audio(sink)?;
        Ok((Capture::Audio(capture), channels))
    }

    fn play_audio(&self, device: &str, render: AudioRender) -> anyhow::Result<(Playback, String)> {
        audio::play_audio(device, render)
    }

    fn capture_browser(
        &self,
        _page: &BrowserPage<'_>,
        _video: VideoSink<Frame>,
        _audio: Option<AudioSink>,
    ) -> anyhow::Result<(Capture, CaptureFormat, Arc<dyn BrowserControl>)> {
        bail!("browser sources aren't supported on Linux")
    }

    fn capture_application_audio(
        &self,
        _app: &str,
        _sink: AudioSink,
    ) -> anyhow::Result<(Capture, u32)> {
        not_yet("application audio capture", "L.6")
    }

    fn create_video_encoder(&self, _spec: &LiveVideoSpec) -> anyhow::Result<VideoEncoder> {
        not_yet("streaming", "L.6")
    }

    fn create_audio_encoder(&self, _spec: &LiveAudioSpec) -> anyhow::Result<AudioEncoder> {
        not_yet("streaming", "L.6")
    }

    fn has_secret(&self, _service: &str, _account: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn secret(&self, _service: &str, _account: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn set_secret(&self, _service: &str, _account: &str, _secret: &str) -> anyhow::Result<()> {
        not_yet("keeping secrets", "L.6")
    }

    fn delete_secret(&self, _service: &str, _account: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn join_files(&self, _spec: &JoinSpec, _progress: OnJoinProgress) -> anyhow::Result<Joined> {
        not_yet("joining takes", "L.6")
    }
}
