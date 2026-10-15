//! Media files written by GStreamer: frames and samples pushed into
//! `appsrc`s, encoded (NVENC where the machine has it, else on the CPU)
//! and muxed into a fragmented MOV or MP4.
//!
//! The muxer writes a `moov` and then a standard fragment (`moof` and its
//! `mdat`) every [`MOVIE_FRAGMENT_SECONDS`], so a crash loses at most the
//! fragment being written (`dash-or-mss`; `first-moov-then-finalise`
//! puts each `mdat` before its `moof`, which leaves a killed file
//! unreadable).

use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use isocast_engine::{FileWriter, MOVIE_FRAGMENT_SECONDS, SAMPLE_RATE, Time, WriterSpec};
use isocast_protocol::{Codec, Container};

use crate::frames::{Frame, Layout};

/// Silent samples the AAC encoder (FFmpeg's) puts before the first real
/// one: exactly its first packet (see [`skip_priming`]).
const AAC_PRIMING: u64 = 1024;
/// Frames waiting for the encoder before more are dropped.
const MAX_QUEUED_FRAMES: u64 = 8;
/// How long finishing may wait for the encoders and muxer to drain.
const FINISH_WAIT: Duration = Duration::from_secs(10);
const NANOS_PER_SEC: i64 = 1_000_000_000;

/// Start GStreamer once per process.
pub(crate) fn init() -> Result<()> {
    static INIT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| gst::init().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow!("could not start GStreamer: {e}"))
}

/// The video encoders tried for `codec`, best first.
fn encoder_candidates(codec: Codec) -> Result<&'static [&'static str]> {
    Ok(match codec {
        Codec::H264 => &["nvh264enc", "vah264enc", "x264enc"],
        Codec::Hevc => &["nvh265enc", "vah265enc", "x265enc"],
        other => {
            bail!("recording.encoder.codec: {other:?} isn't available on Linux; valid: h264, hevc")
        }
    })
}

/// The first video encoder for `codec` this machine can make.
fn make_encoder(codec: Codec, quality: u8, fps: u32) -> Result<(gst::Element, &'static str)> {
    let candidates = encoder_candidates(codec)?;
    for &name in candidates {
        let Ok(encoder) = gst::ElementFactory::make(name).build() else {
            continue;
        };
        configure_encoder(&encoder, name, quality, fps);
        return Ok((encoder, name));
    }
    bail!(
        "no {codec:?} encoder found (tried {}); install the GStreamer plugins for one",
        candidates.join(", ")
    )
}

/// A quantiser for the 1–100 quality scale (VideoToolbox's): 100 is near
/// lossless (15), the default "high" (70) about 22, 30 about 32.
fn quantiser(quality: u8) -> u32 {
    40 - u32::from(quality.min(100)) / 4
}

fn configure_encoder(encoder: &gst::Element, name: &str, quality: u8, fps: u32) {
    let qp = quantiser(quality);
    // A keyframe every second or two keeps fragments independently
    // decodable and seeking quick.
    let gop = (fps * 2).max(1) as i32;
    match name {
        "nvh264enc" | "nvh265enc" => {
            encoder.set_property_from_str("rc-mode", "vbr");
            encoder.set_property("const-quality", f64::from(qp));
            encoder.set_property("bitrate", 0u32);
            encoder.set_property("gop-size", gop);
            encoder.set_property("bframes", 0u32);
            encoder.set_property_from_str("preset", "p4");
        }
        "vah264enc" | "vah265enc" => {
            encoder.set_property_from_str("rate-control", "cqp");
            encoder.set_property("qpi", qp);
            encoder.set_property("qpp", qp);
            encoder.set_property("key-int-max", gop as u32);
            encoder.set_property("b-frames", 0u32);
        }
        "x264enc" => {
            // Constant quality (CRF), fast enough for real time.
            encoder.set_property_from_str("pass", "qual");
            encoder.set_property("quantizer", qp);
            encoder.set_property_from_str("speed-preset", "veryfast");
            encoder.set_property("key-int-max", gop as u32);
            encoder.set_property("bframes", 0u32);
        }
        "x265enc" => {
            encoder.set_property("qp", qp as i32);
            encoder.set_property_from_str("speed-preset", "veryfast");
            encoder.set_property("key-int-max", gop);
        }
        _ => {}
    }
}

/// Drop the AAC encoder's first packet, its priming, and move the rest
/// that much earlier, so the audio lines up with the video. `qtmux`
/// writes no edit list for priming (it ignores `GstAudioClippingMeta`,
/// and fragmented files get none), and this works in a file cut short by
/// a crash too. The cost: without the dropped packet's overlap the first
/// 21 ms decode as a short fade-in, as after any cut without re-encoding.
fn skip_priming(parse: &gst::Element) {
    let priming =
        gst::ClockTime::from_nseconds(AAC_PRIMING * NANOS_PER_SEC as u64 / u64::from(SAMPLE_RATE));
    let first = std::sync::atomic::AtomicBool::new(true);
    let pad = parse.static_pad("src").expect("aacparse has a src pad");
    pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
        if first.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return gst::PadProbeReturn::Drop;
        }
        if let Some(gst::PadProbeData::Buffer(buffer)) = info.data.as_mut() {
            let buffer = buffer.make_mut();
            if let Some(pts) = buffer.pts() {
                buffer.set_pts(pts.saturating_sub(priming));
            }
            if let Some(dts) = buffer.dts() {
                buffer.set_dts(dts.saturating_sub(priming));
            }
        }
        gst::PadProbeReturn::Ok
    });
}

fn make(name: &str) -> Result<gst::Element> {
    gst::ElementFactory::make(name)
        .build()
        .with_context(|| format!("the GStreamer element {name} is missing"))
}

fn video_caps(layout: Layout, width: u32, height: u32, fps: u32) -> gst::Caps {
    let format = match layout {
        Layout::Bgra => "BGRA",
        Layout::Nv12 => "NV12",
    };
    gst::Caps::builder("video/x-raw")
        .field("format", format)
        .field("width", width as i32)
        .field("height", height as i32)
        .field("framerate", gst::Fraction::new(fps as i32, 1))
        .field("colorimetry", "bt709")
        .field("interlace-mode", "progressive")
        .build()
}

pub struct Writer {
    pipeline: gst::Pipeline,
    video: gst_app::AppSrc,
    audio: Vec<(gst_app::AppSrc, u32)>,
    path: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    start: Option<Time>,
    layout: Option<Layout>,
    /// The last frame appended and when it shows, repeated at the end if
    /// the timeline runs past it.
    last: Option<(Frame, Time)>,
    finished: bool,
}

impl Writer {
    pub fn create(spec: &WriterSpec) -> Result<Self> {
        init()?;
        let video_spec = &spec.video;
        if video_spec.encoder.bit_depth != 8 {
            bail!(
                "recording.encoder.bit_depth: {}-bit isn't supported on Linux yet; valid: 8",
                video_spec.encoder.bit_depth
            );
        }
        if spec.path.exists() {
            bail!("{} already exists", spec.path.display());
        }
        let pipeline = gst::Pipeline::with_name("writer");
        let mux = make(match spec.container {
            Container::Mov => "qtmux",
            Container::Mp4 => "mp4mux",
        })?;
        mux.set_property("fragment-duration", (MOVIE_FRAGMENT_SECONDS * 1000) as u32);
        mux.set_property_from_str("fragment-mode", "dash-or-mss");
        let sink = make("filesink")?;
        sink.set_property("location", spec.path.to_string_lossy().as_ref());
        sink.set_property("sync", false);
        pipeline.add_many([&mux, &sink])?;
        mux.link(&sink)?;

        let video = gst_app::AppSrc::builder()
            .name("video")
            .format(gst::Format::Time)
            .is_live(false)
            .build();
        // Back-pressure is ours (`MAX_QUEUED_FRAMES`), so appsrc never
        // drops or blocks on its own.
        video.set_max_bytes(0);
        let convert = make("videoconvert")?;
        let queue = make("queue")?;
        let (encoder, encoder_name) = make_encoder(
            video_spec.encoder.codec,
            video_spec.encoder.quality.value(),
            video_spec.fps,
        )?;
        let parse = make(match video_spec.encoder.codec {
            Codec::H264 => "h264parse",
            _ => "h265parse",
        })?;
        pipeline.add_many([video.upcast_ref(), &convert, &queue, &encoder, &parse])?;
        gst::Element::link_many([video.upcast_ref(), &convert, &queue, &encoder, &parse, &mux])
            .with_context(|| format!("could not connect {encoder_name} to the muxer"))?;

        let mut audio = Vec::new();
        for (index, track) in spec.audio.iter().enumerate() {
            let src = gst_app::AppSrc::builder()
                .name(format!("audio{index}"))
                .format(gst::Format::Time)
                .is_live(false)
                .caps(
                    &gst::Caps::builder("audio/x-raw")
                        .field("format", "F32LE")
                        .field("layout", "interleaved")
                        .field("rate", SAMPLE_RATE as i32)
                        .field("channels", track.channels as i32)
                        .build(),
                )
                .build();
            src.set_max_bytes(0);
            let convert = make("audioconvert")?;
            let encode = make("avenc_aac").context("AAC needs gst-libav installed")?;
            let parse = make("aacparse")?;
            skip_priming(&parse);
            let queue = make("queue")?;
            pipeline.add_many([src.upcast_ref(), &convert, &encode, &parse, &queue])?;
            gst::Element::link_many([src.upcast_ref(), &convert, &encode, &parse, &queue, &mux])
                .context("could not connect the AAC encoder to the muxer")?;
            audio.push((src, track.channels));
        }

        pipeline
            .set_state(gst::State::Playing)
            .context("could not start the file writer")?;
        let writer = Self {
            pipeline,
            video,
            audio,
            path: spec.path.clone(),
            width: video_spec.width,
            height: video_spec.height,
            fps: video_spec.fps,
            start: None,
            layout: None,
            last: None,
            finished: false,
        };
        writer.check_errors()?;
        Ok(writer)
    }

    /// The first error the pipeline reported, if any.
    fn check_errors(&self) -> Result<()> {
        let bus = self.pipeline.bus().expect("pipelines have a bus");
        while let Some(message) = bus.pop_filtered(&[gst::MessageType::Error]) {
            if let gst::MessageView::Error(err) = message.view() {
                bail!(
                    "writing {} failed: {} ({})",
                    self.path.display(),
                    err.error(),
                    err.debug().unwrap_or_default()
                );
            }
        }
        Ok(())
    }

    /// `pts` on the file's timeline, in nanoseconds.
    fn file_time(&self, pts: Time) -> Result<i64> {
        let start = self.start.context("the writer was not started")?;
        Ok(pts.as_nanos() - start.as_nanos())
    }

    /// Frames waiting in the video appsrc.
    fn queued(&self) -> u64 {
        self.video.property::<u64>("current-level-buffers")
    }

    fn frame_duration(&self) -> i64 {
        NANOS_PER_SEC / i64::from(self.fps)
    }

    fn push_frame(&mut self, frame: &Frame, at: i64) -> Result<()> {
        if (frame.width(), frame.height()) != (self.width, self.height) {
            bail!(
                "a {}×{} frame can't go in a {}×{} file",
                frame.width(),
                frame.height(),
                self.width,
                self.height
            );
        }
        if self.layout != Some(frame.layout()) {
            self.video.set_caps(Some(&video_caps(
                frame.layout(),
                self.width,
                self.height,
                self.fps,
            )));
            self.layout = Some(frame.layout());
        }
        let mut buffer = gst::Buffer::from_mut_slice(frame.pixels()?);
        let buffer_mut = buffer.get_mut().expect("a new buffer is writable");
        buffer_mut.set_pts(gst::ClockTime::from_nseconds(at as u64));
        buffer_mut.set_duration(gst::ClockTime::from_nseconds(self.frame_duration() as u64));
        self.video
            .push_buffer(buffer)
            .map_err(|e| anyhow!("the video encoder refused a frame: {e:?}"))?;
        Ok(())
    }
}

impl FileWriter for Writer {
    type Frame = Frame;

    fn start(&mut self, at: Time) -> Result<()> {
        if self.start.is_some() {
            bail!("the writer was already started");
        }
        self.start = Some(at);
        Ok(())
    }

    fn append_video(&mut self, frame: &Frame, pts: Time) -> Result<bool> {
        self.check_errors()?;
        let at = self.file_time(pts)?;
        if at < 0 || self.last.as_ref().is_some_and(|(_, last)| pts <= *last) {
            return Ok(false);
        }
        // Wait up to a frame's time for a busy encoder, then drop.
        let deadline = Instant::now() + Duration::from_nanos(self.frame_duration() as u64);
        while self.queued() >= MAX_QUEUED_FRAMES {
            if Instant::now() >= deadline {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.push_frame(frame, at)?;
        self.last = Some((frame.clone(), pts));
        Ok(true)
    }

    fn queue_depth(&self) -> usize {
        self.queued() as usize
    }

    fn append_audio(&mut self, track: usize, samples: &[f32], pts: Time) -> Result<bool> {
        self.check_errors()?;
        let at = self.file_time(pts)?;
        let (src, channels) = self
            .audio
            .get(track)
            .with_context(|| format!("the file has no audio track {track}"))?;
        let channels = *channels as usize;
        // Samples from before the file's start are cut off.
        let skip = if at < 0 {
            let nanos = (-at) as u128 * u128::from(SAMPLE_RATE);
            nanos.div_ceil(NANOS_PER_SEC as u128) as usize * channels
        } else {
            0
        };
        let Some(samples) = samples.get(skip..).filter(|s| !s.is_empty()) else {
            return Ok(true);
        };
        let at = at.max(0) + (skip / channels) as i64 * NANOS_PER_SEC / i64::from(SAMPLE_RATE);
        let frames = (samples.len() / channels) as i64;
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut buffer = gst::Buffer::from_mut_slice(bytes);
        let buffer_mut = buffer.get_mut().expect("a new buffer is writable");
        buffer_mut.set_pts(gst::ClockTime::from_nseconds(at as u64));
        buffer_mut.set_duration(gst::ClockTime::from_nseconds(
            (frames * NANOS_PER_SEC / i64::from(SAMPLE_RATE)) as u64,
        ));
        src.push_buffer(buffer)
            .map_err(|e| anyhow!("the audio encoder refused samples: {e:?}"))?;
        Ok(true)
    }

    fn finish(mut self, end: Time) -> Result<u64> {
        self.check_errors()?;
        let end_at = self.file_time(end)?;
        let last_slot = end_at - self.frame_duration();
        let mut repeated = 0;
        // Hold the last picture to the end of the timeline.
        if let Some((frame, pts)) = self.last.take()
            && self.file_time(pts)? + self.frame_duration() / 2 < last_slot
        {
            self.push_frame(&frame, last_slot)?;
            repeated = 1;
        }
        self.video
            .end_of_stream()
            .map_err(|e| anyhow!("could not end the video: {e:?}"))?;
        for (src, _) in &self.audio {
            src.end_of_stream()
                .map_err(|e| anyhow!("could not end the audio: {e:?}"))?;
        }
        let bus = self.pipeline.bus().expect("pipelines have a bus");
        let deadline = Instant::now() + FINISH_WAIT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                bail!(
                    "the file encoder did not drain within 10 seconds: {}",
                    self.path.display()
                );
            }
            let Some(message) = bus.timed_pop_filtered(
                gst::ClockTime::from_nseconds(left.as_nanos() as u64),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            ) else {
                continue;
            };
            match message.view() {
                gst::MessageView::Eos(_) => break,
                gst::MessageView::Error(err) => bail!(
                    "finishing {} failed: {} ({})",
                    self.path.display(),
                    err.error(),
                    err.debug().unwrap_or_default()
                ),
                _ => {}
            }
        }
        self.pipeline
            .set_state(gst::State::Null)
            .context("could not close the file")?;
        self.finished = true;
        Ok(repeated)
    }

    fn cancel(mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
        self.finished = true;
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.pipeline.set_state(gst::State::Null);
        }
    }
}
