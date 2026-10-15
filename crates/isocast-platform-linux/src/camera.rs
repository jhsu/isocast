//! Cameras through V4L2 (GStreamer's `v4l2src`), converted to NV12.
//!
//! PipeWire also lists cameras, but offers them as DMA-BUF only, which
//! needs the zero-copy import that doesn't exist yet; V4L2 is what OBS
//! uses too.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use isocast_engine::{CameraRequest, CaptureFormat, VideoSink};
use isocast_protocol::Canvas;
use isocast_protocol::devices::{Camera, CameraKind, FrameRateRange, VideoFormat};

use crate::frames::{Frame, Layout};
use crate::pipeline::{Running, make};

/// Raw formats `videoconvert` turns into NV12, and their V4L2 four-character
/// codes. Others (like an infrared camera's GREY) aren't offered.
const RAW_FORMATS: &[(&str, &str)] = &[
    ("NV12", "NV12"),
    ("YUY2", "YUYV"),
    ("UYVY", "UYVY"),
    ("I420", "YU12"),
    ("YV12", "YV12"),
    ("NV21", "NV21"),
    ("BGRx", "XR24"),
    ("BGR", "BGR3"),
    ("RGB", "RGB3"),
];

/// One way a camera can deliver frames: a format at a size, and its rates.
#[derive(Debug, Clone)]
struct Mode {
    /// `image/jpeg`, or the raw format's GStreamer name.
    format: Option<&'static str>,
    fourcc: &'static str,
    width: u32,
    height: u32,
    /// Discrete rates, as the device states them.
    rates: Vec<gst::Fraction>,
    /// Continuous ranges, if it offers any.
    ranges: Vec<(gst::Fraction, gst::Fraction)>,
}

impl Mode {
    /// Formats that need no decoding are cheaper: NV12, then other raw
    /// ones, then MJPEG.
    fn pixel_score(&self) -> u8 {
        match self.format {
            Some("NV12") => 2,
            Some(_) => 1,
            None => 0,
        }
    }

    fn caps(&self, rate: gst::Fraction) -> gst::Caps {
        let builder = match self.format {
            Some(format) => gst::Caps::builder("video/x-raw").field("format", format),
            None => gst::Caps::builder("image/jpeg"),
        };
        builder
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", rate)
            .build()
    }
}

fn rate(fraction: gst::Fraction) -> f64 {
    f64::from(fraction.numer()) / f64::from(fraction.denom().max(1))
}

/// Every mode in a device's caps.
fn modes(caps: &gst::Caps) -> Vec<Mode> {
    let mut modes = Vec::new();
    for structure in caps.iter() {
        let (format, fourcc) = match structure.name().as_str() {
            "image/jpeg" => (None, "MJPG"),
            "video/x-raw" => {
                let Ok(name) = structure.get::<&str>("format") else {
                    continue;
                };
                let Some(&(format, fourcc)) = RAW_FORMATS.iter().find(|(f, _)| *f == name) else {
                    continue;
                };
                (Some(format), fourcc)
            }
            _ => continue,
        };
        let (Ok(width), Ok(height)) = (
            structure.get::<i32>("width"),
            structure.get::<i32>("height"),
        ) else {
            continue;
        };
        let mut rates = Vec::new();
        let mut ranges = Vec::new();
        if let Ok(fraction) = structure.get::<gst::Fraction>("framerate") {
            rates.push(fraction);
        } else if let Ok(list) = structure.get::<gst::List>("framerate") {
            rates.extend(list.iter().filter_map(|v| v.get::<gst::Fraction>().ok()));
        } else if let Ok(range) = structure.get::<gst::FractionRange>("framerate") {
            ranges.push((range.min(), range.max()));
        }
        rates.retain(|r| r.numer() > 0);
        rates.sort_by(|a, b| rate(*a).total_cmp(&rate(*b)));
        if rates.is_empty() && ranges.is_empty() {
            continue;
        }
        modes.push(Mode {
            format,
            fourcc,
            width: width as u32,
            height: height as u32,
            rates,
            ranges,
        });
    }
    modes
}

/// A camera: its device and what setups know it by.
struct Found {
    path: PathBuf,
    id: String,
    name: String,
    kind: CameraKind,
    modes: Vec<Mode>,
}

/// The stable name udev gives a device under /dev/v4l/by-id, if any.
fn by_id(path: &Path) -> Option<String> {
    let target = path.canonicalize().ok()?;
    std::fs::read_dir("/dev/v4l/by-id")
        .ok()?
        .flatten()
        .find(|entry| entry.path().canonicalize().ok().as_deref() == Some(&target))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
}

fn find_all() -> Result<Vec<Found>> {
    crate::writer::init()?;
    let provider = gst::DeviceProviderFactory::by_name("v4l2deviceprovider")
        .context("cameras need GStreamer's video4linux2 plugin")?;
    let mut cameras: Vec<Found> = Vec::new();
    for device in provider.devices() {
        let properties = device.properties();
        let string = |key: &str| {
            properties
                .as_ref()
                .and_then(|p| p.get::<String>(key).ok())
                .unwrap_or_default()
        };
        let path = PathBuf::from(string("device.path"));
        let Some(caps) = device.caps() else {
            continue;
        };
        let modes = modes(&caps);
        if modes.is_empty() || path.as_os_str().is_empty() {
            continue;
        }
        let card = string("v4l2.device.card");
        let name = if card.is_empty() {
            device.display_name().to_string()
        } else {
            card
        };
        let driver = string("v4l2.device.driver");
        let lower = name.to_lowercase();
        let kind = if driver.contains("loopback") {
            CameraKind::Virtual
        } else if ["integrated", "built-in", "internal"]
            .iter()
            .any(|w| lower.contains(w))
        {
            CameraKind::BuiltIn
        } else if string("v4l2.device.bus_info").starts_with("usb") {
            CameraKind::External
        } else {
            CameraKind::Other
        };
        // Two of the same camera get told apart by number.
        let same = cameras.iter().filter(|c| c.name.starts_with(&name)).count();
        let name = if same == 0 {
            name
        } else {
            format!("{name} ({})", same + 1)
        };
        cameras.push(Found {
            id: by_id(&path).unwrap_or_else(|| path.to_string_lossy().into_owned()),
            path,
            name,
            kind,
            modes,
        });
    }
    Ok(cameras)
}

/// The camera `query` names: its id, path or exact name, else its name in
/// any case, else the shortest name containing it ("default": the first).
fn find(query: &str) -> Result<Found> {
    let mut cameras = find_all()?;
    let lower = query.to_lowercase();
    let index = if query == "default" {
        (!cameras.is_empty()).then_some(0)
    } else {
        cameras
            .iter()
            .position(|c| c.id == query || c.name == query || c.path == Path::new(query))
            .or_else(|| cameras.iter().position(|c| c.name.to_lowercase() == lower))
            .or_else(|| {
                cameras
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.name.to_lowercase().contains(&lower))
                    .min_by_key(|(_, c)| c.name.len())
                    .map(|(i, _)| i)
            })
    };
    match index {
        Some(index) => Ok(cameras.swap_remove(index)),
        None => {
            let names: Vec<String> = cameras
                .iter()
                .map(|c| format!("{:?} (id {})", c.name, c.id))
                .collect();
            Err(anyhow!(
                "no camera matches {query:?}; valid: \"default\", {}",
                names.join(", ")
            ))
        }
    }
}

pub(crate) fn cameras() -> Result<Vec<Camera>> {
    Ok(find_all()?
        .into_iter()
        .map(|found| {
            let mut formats: Vec<VideoFormat> = found
                .modes
                .iter()
                .map(|mode| VideoFormat {
                    width: mode.width,
                    height: mode.height,
                    pixel_format: mode.fourcc.into(),
                    frame_rates: mode.rates.iter().map(|r| rate(*r)).collect(),
                    frame_rate_ranges: mode
                        .ranges
                        .iter()
                        .map(|(min, max)| FrameRateRange {
                            min: rate(*min),
                            max: rate(*max),
                        })
                        .rev()
                        .collect(),
                })
                .collect();
            formats.sort_by_key(|f| std::cmp::Reverse(u64::from(f.width) * u64::from(f.height)));
            Camera {
                id: found.id,
                model: found.name.clone(),
                name: found.name,
                manufacturer: String::new(),
                kind: found.kind,
                formats,
                suspended: false,
            }
        })
        .collect())
}

/// The mode and rate to capture in, as on macOS. Without a request: the
/// highest frame rate up to the canvas rate, then the largest size that
/// fits on the canvas either way round (else the smallest larger one),
/// turned as the canvas is if it can be, then a format that needs no
/// decoding.
fn choose(
    modes: &[Mode],
    request: CameraRequest,
    canvas: &Canvas,
) -> Result<(Mode, gst::Fraction)> {
    let target = f64::from(request.fps.unwrap_or(canvas.fps));
    // Frame rate, fits, size, turned as the canvas, pixel format.
    type Score = (u32, bool, u64, bool, u8);
    let mut best: Option<(Score, Mode, gst::Fraction)> = None;
    let mut offered = Vec::new();
    for mode in modes {
        let (width, height) = (mode.width, mode.height);
        if request.size.is_some_and(|wanted| wanted != (width, height)) {
            offered.push(format!("{width}×{height}"));
            continue;
        }
        // The rate closest to the target, by preference not above it.
        let mut candidates: Vec<gst::Fraction> = mode.rates.clone();
        for &(min, max) in &mode.ranges {
            let wanted = gst::Fraction::approximate_f64(target).unwrap_or(max);
            candidates.push(if target < rate(min) {
                min
            } else if target > rate(max) {
                max
            } else {
                wanted
            });
        }
        let Some(chosen) = candidates.into_iter().min_by(|a, b| {
            let key = |r: &gst::Fraction| {
                let diff = rate(*r) - target;
                (diff > 0.5, diff.abs())
            };
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            continue;
        };
        offered.push(format!("{width}×{height} @ {:.2}", rate(chosen)));
        if request.fps.is_some() && (rate(chosen) - target).abs() > 0.5 {
            continue;
        }
        let fps = rate(chosen).round() as u32;
        let area = u64::from(width) * u64::from(height);
        let fits = width.max(height) <= canvas.width.max(canvas.height)
            && width.min(height) <= canvas.width.min(canvas.height);
        let score = (
            fps.min(canvas.fps),
            fits,
            if fits { area } else { u64::MAX - area },
            (width >= height) == (canvas.width >= canvas.height),
            mode.pixel_score(),
        );
        if best.as_ref().is_none_or(|(b, ..)| score > *b) {
            best = Some((score, mode.clone(), chosen));
        }
    }
    // Each size once, however many formats offer it.
    let mut seen = std::collections::HashSet::new();
    offered.retain(|entry| seen.insert(entry.clone()));
    best.map(|(_, mode, rate)| (mode, rate)).ok_or_else(|| {
        let wanted = match (request.size, request.fps) {
            (Some((w, h)), Some(fps)) => format!("{w}×{h} @ {fps}"),
            (Some((w, h)), None) => format!("{w}×{h}"),
            (None, Some(fps)) => format!("{fps} fps"),
            (None, None) => "any format".into(),
        };
        anyhow!(
            "no format matches {wanted}; the device offers: {}",
            offered.join(", ")
        )
    })
}

/// The camera a source would record and the format, without opening it.
pub(crate) fn probe(
    query: &str,
    request: CameraRequest,
    canvas: &Canvas,
) -> Result<(String, CaptureFormat)> {
    let found = find(query)?;
    let (mode, chosen) = choose(&found.modes, request, canvas)?;
    Ok((
        found.name,
        CaptureFormat {
            width: mode.width,
            height: mode.height,
            fps: rate(chosen).round() as u32,
        },
    ))
}

/// Copy a mapped NV12 frame's planes, rows packed tightly.
fn packed_nv12(frame: &gst_video::VideoFrameRef<&gst::BufferRef>) -> Result<Vec<u8>> {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let mut pixels = Vec::with_capacity(Layout::Nv12.len(width as u32, height as u32));
    for (plane, rows) in [(0, height), (1, height.div_ceil(2))] {
        let stride = frame.plane_stride()[plane] as usize;
        let data = frame.plane_data(plane as u32)?;
        let row = if plane == 0 {
            width
        } else {
            width.div_ceil(2) * 2
        };
        for y in 0..rows {
            pixels.extend_from_slice(&data[y * stride..y * stride + row]);
        }
    }
    Ok(pixels)
}

/// Start capturing the camera `query` names in the format `request` asks
/// for, or its best one for `canvas`.
pub fn capture_camera(
    query: &str,
    request: CameraRequest,
    canvas: &Canvas,
    sink: VideoSink<Frame>,
) -> Result<(Running, CaptureFormat)> {
    let found = find(query)?;
    let (mode, chosen) = choose(&found.modes, request, canvas)?;
    if mode.width % 2 != 0 || mode.height % 2 != 0 {
        bail!(
            "{}×{} has an odd side, which 4:2:0 can't hold",
            mode.width,
            mode.height
        );
    }
    let pipeline = gst::Pipeline::new();
    let source = make("v4l2src")?;
    source.set_property("device", found.path.to_string_lossy().as_ref());
    let filter = make("capsfilter")?;
    filter.set_property("caps", mode.caps(chosen));
    let mut elements = vec![source, filter];
    if mode.format.is_none() {
        elements.push(make("jpegdec")?);
    }
    elements.push(make("videoconvert")?);
    let appsink = gst_app::AppSink::builder()
        .caps(
            &gst::Caps::builder("video/x-raw")
                .field("format", "NV12")
                .field("colorimetry", "bt709")
                .build(),
        )
        .sync(false)
        // A slow consumer gets the newest frame, not a growing queue.
        .max_buffers(2)
        .drop(true)
        .build();
    pipeline.add_many(&elements)?;
    pipeline.add(&appsink)?;
    elements.push(appsink.clone().upcast());
    gst::Element::link_many(&elements)
        .with_context(|| format!("could not open {:?} in {}", found.name, mode.fourcc))?;

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
                let info =
                    gst_video::VideoInfo::from_caps(caps).map_err(|_| gst::FlowError::Error)?;
                let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info)
                    .map_err(|_| gst::FlowError::Error)?;
                let pixels = packed_nv12(&frame).map_err(|_| gst::FlowError::Error)?;
                let frame = Frame::from_pixels(info.width(), info.height(), Layout::Nv12, pixels)
                    .map_err(|_| gst::FlowError::Error)?;
                frames(frame, captured);
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    let format = CaptureFormat {
        width: mode.width,
        height: mode.height,
        fps: rate(chosen).round() as u32,
    };
    let running = Running::start(pipeline, format!("camera {:?}", found.name), sink.stopped)?;
    Ok((running, format))
}
