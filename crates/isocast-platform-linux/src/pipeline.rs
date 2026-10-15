//! Running GStreamer pipelines for captures and playback.
//!
//! Pipelines run on GStreamer's system clock, which is `CLOCK_MONOTONIC`
//! like the engine's, so a buffer's capture time is the pipeline's base
//! time plus its running time.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use isocast_engine::Time;

pub(crate) fn make(name: &str) -> Result<gst::Element> {
    crate::writer::init()?;
    gst::ElementFactory::make(name)
        .build()
        .with_context(|| format!("the GStreamer element {name} is missing"))
}

/// A new pipeline, GStreamer started first.
pub(crate) fn new() -> Result<gst::Pipeline> {
    crate::writer::init()?;
    Ok(gst::Pipeline::new())
}

/// When `buffer` (from a live source in `pipeline`) was captured, on the
/// engine's clock.
pub(crate) fn capture_time(pipeline: &gst::Pipeline, buffer: &gst::BufferRef) -> Option<Time> {
    let pts = buffer.pts()?;
    let base = pipeline.base_time()?;
    Some(Time::from_nanos((base + pts).nseconds() as i64))
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
    pub(crate) fn start(
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
