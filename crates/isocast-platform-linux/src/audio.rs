//! Audio in and out through GStreamer: microphones and the system's sound
//! (the default output's monitor) captured as interleaved `f32` at
//! [`SAMPLE_RATE`], and monitoring played on an output device.
//!
//! Pipelines run on GStreamer's system clock, which is `CLOCK_MONOTONIC`
//! like the engine's, so a buffer's capture time is the pipeline's base
//! time plus its running time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use isocast_engine::{AudioRender, AudioSink, SAMPLE_RATE, Time};

use crate::devices;

/// Samples handed to an output device at a time: 10 ms.
const PLAYBACK_CHUNK: usize = SAMPLE_RATE as usize / 100;

fn make(name: &str) -> Result<gst::Element> {
    crate::writer::init()?;
    gst::ElementFactory::make(name)
        .build()
        .with_context(|| format!("the GStreamer element {name} is missing"))
}

fn raw_caps(channels: u32) -> gst::Caps {
    gst::Caps::builder("audio/x-raw")
        .field("format", "F32LE")
        .field("layout", "interleaved")
        .field("rate", SAMPLE_RATE as i32)
        .field("channels", channels as i32)
        .build()
}

/// A running pipeline, and the thread watching it for errors. Dropping it
/// stops both.
pub struct Running {
    pipeline: gst::Pipeline,
    stop: Arc<AtomicBool>,
    watch: Option<JoinHandle<()>>,
}

impl Running {
    /// Start `pipeline` on the monotonic clock, calling `stopped` if it
    /// fails or ends on its own.
    fn start(
        pipeline: gst::Pipeline,
        what: String,
        stopped: Box<dyn Fn(String) + Send + Sync>,
    ) -> Result<Self> {
        pipeline.use_clock(Some(&gst::SystemClock::obtain()));
        pipeline
            .set_state(gst::State::Playing)
            .with_context(|| format!("could not start {what}"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let bus = pipeline.bus().expect("pipelines have a bus");
        let watch = {
            let stop = stop.clone();
            thread::Builder::new()
                .name(format!("isocast {what}"))
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let Some(message) = bus.timed_pop_filtered(
                            gst::ClockTime::from_mseconds(100),
                            &[gst::MessageType::Error, gst::MessageType::Eos],
                        ) else {
                            continue;
                        };
                        match message.view() {
                            gst::MessageView::Error(err) => {
                                stopped(format!("{what} failed: {}", err.error()));
                            }
                            _ => stopped(format!("{what} ended")),
                        }
                        return;
                    }
                })?
        };
        Ok(Self {
            pipeline,
            stop,
            watch: Some(watch),
        })
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.pipeline.set_state(gst::State::Null);
        if let Some(watch) = self.watch.take() {
            let _ = watch.join();
        }
    }
}

/// Capture from `source`, converted to `channels` interleaved `f32` at
/// [`SAMPLE_RATE`], into `sink`.
fn capture(source: gst::Element, channels: u32, sink: AudioSink, what: String) -> Result<Running> {
    let pipeline = gst::Pipeline::new();
    let convert = make("audioconvert")?;
    let resample = make("audioresample")?;
    let appsink = gst_app::AppSink::builder()
        .caps(&raw_caps(channels))
        .sync(false)
        .build();
    pipeline.add_many([&source, &convert, &resample, appsink.upcast_ref()])?;
    gst::Element::link_many([&source, &convert, &resample, appsink.upcast_ref()])
        .with_context(|| format!("could not connect {what}"))?;
    let samples = sink.samples;
    let clock_of = pipeline.downgrade();
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let (Some(buffer), Some(pipeline)) = (sample.buffer(), clock_of.upgrade()) else {
                    return Ok(gst::FlowSuccess::Ok);
                };
                let (Some(pts), Some(base)) = (buffer.pts(), pipeline.base_time()) else {
                    return Ok(gst::FlowSuccess::Ok);
                };
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                let values: Vec<f32> = map
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect();
                let captured = Time::from_nanos((base + pts).nseconds() as i64);
                samples(&values, captured);
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    Running::start(pipeline, what, sink.stopped)
}

/// Start capturing the microphone `query` names ("default" for the default
/// input). Returns the capture and its channel count.
pub fn capture_microphone(query: &str, sink: AudioSink) -> Result<(Running, u32)> {
    let found = devices::find("Audio/Source", "microphone", query)?;
    let channels = devices::channels(&found.device);
    let source = found
        .device
        .create_element(None)
        .with_context(|| format!("could not open {:?}", found.name))?;
    let running = capture(
        source,
        channels,
        sink,
        format!("microphone {:?}", found.name),
    )?;
    Ok((running, channels))
}

/// Start capturing what the default output plays, through its monitor
/// (following the default when it changes). It includes this process's
/// own monitoring: `exclude_own` isn't supported yet.
pub fn capture_system_audio(sink: AudioSink) -> Result<(Running, u32)> {
    let source = make("pulsesrc").context("system audio needs PulseAudio or pipewire-pulse")?;
    source.set_property("device", "@DEFAULT_MONITOR@");
    let running = capture(source, 2, sink, "system audio".into())?;
    Ok((running, 2))
}

/// Start playing what `render` makes on the output `query` names
/// ("default" for the default output). Returns the playback and the
/// device's name.
pub fn play_audio(query: &str, render: AudioRender) -> Result<(Running, String)> {
    let found = devices::find("Audio/Sink", "audio output", query)?;
    let output = found
        .device
        .create_element(None)
        .with_context(|| format!("could not open {:?}", found.name))?;
    let pipeline = gst::Pipeline::new();
    let appsrc = gst_app::AppSrc::builder()
        .caps(&raw_caps(2))
        .format(gst::Format::Time)
        // A few chunks ahead: enough to ride out scheduling, little delay.
        .max_bytes((PLAYBACK_CHUNK * 2 * 4 * 4) as u64)
        .build();
    let convert = make("audioconvert")?;
    let resample = make("audioresample")?;
    pipeline.add_many([appsrc.upcast_ref(), &convert, &resample, &output])?;
    gst::Element::link_many([appsrc.upcast_ref(), &convert, &resample, &output])
        .with_context(|| format!("could not connect {:?}", found.name))?;
    let render = Mutex::new((render, 0u64));
    appsrc.set_callbacks(
        gst_app::AppSrcCallbacks::builder()
            .need_data(move |appsrc, _| {
                let mut guard = render.lock().unwrap_or_else(|e| e.into_inner());
                let (render, played) = &mut *guard;
                let mut samples = vec![0f32; PLAYBACK_CHUNK * 2];
                render(&mut samples);
                let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                let mut buffer = gst::Buffer::from_mut_slice(bytes);
                let to_time = |frames: u64| {
                    gst::ClockTime::from_nseconds(frames * 1_000_000_000 / u64::from(SAMPLE_RATE))
                };
                let buffer_mut = buffer.get_mut().expect("a new buffer is writable");
                buffer_mut.set_pts(to_time(*played));
                buffer_mut.set_duration(to_time(PLAYBACK_CHUNK as u64));
                *played += PLAYBACK_CHUNK as u64;
                let _ = appsrc.push_buffer(buffer);
            })
            .build(),
    );
    let name = found.name;
    let running = Running::start(pipeline, format!("playback on {name:?}"), Box::new(|_| {}))
        .map_err(|e| anyhow!("{e:#}"))?;
    Ok((running, name))
}
