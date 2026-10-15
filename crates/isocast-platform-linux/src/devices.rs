//! Devices, as GStreamer's device monitor sees them: PipeWire's where it
//! runs, else PulseAudio's or ALSA's.

use anyhow::{Result, anyhow};
use gstreamer as gst;
use gstreamer::prelude::*;
use isocast_protocol::devices::{AudioOutput, Microphone};

/// A device and what setups know it by.
pub(crate) struct Found {
    pub device: gst::Device,
    /// Stable across reboots: PipeWire's `node.name` where there is one.
    pub id: String,
    pub name: String,
    pub default: bool,
}

/// The devices of `class` ("Audio/Source", "Audio/Sink", "Video/Source").
/// Only that exact class: PipeWire also lists raw ALSA nodes as
/// `…/Internal`, duplicates of the devices users pick.
pub(crate) fn find_all(class: &str) -> Result<Vec<Found>> {
    crate::writer::init()?;
    let monitor = gst::DeviceMonitor::new();
    monitor.add_filter(Some(class), None);
    monitor
        .start()
        .map_err(|_| anyhow!("could not list devices ({class})"))?;
    let devices = monitor.devices();
    monitor.stop();
    Ok(devices
        .into_iter()
        .filter(|device| device.device_class() == class)
        .map(|device| {
            let properties = device.properties();
            let string = |key: &str| {
                properties
                    .as_ref()
                    .and_then(|p| p.get::<String>(key).ok())
                    .filter(|s| !s.is_empty())
            };
            let name = device.display_name().to_string();
            Found {
                id: string("node.name")
                    .or_else(|| string("device.name"))
                    .unwrap_or_else(|| name.clone()),
                default: properties
                    .as_ref()
                    .and_then(|p| p.get::<bool>("is-default").ok())
                    .unwrap_or(false),
                name,
                device,
            }
        })
        .collect())
}

/// The device `query` names: its id or exact name, else its name in any
/// case, else the shortest name containing it ("default": the default).
/// An error lists the valid choices.
pub(crate) fn find(class: &str, what: &str, query: &str) -> Result<Found> {
    let mut devices = find_all(class)?;
    let lower = query.to_lowercase();
    let index = if query == "default" {
        devices
            .iter()
            .position(|d| d.default)
            .or((!devices.is_empty()).then_some(0))
    } else {
        devices
            .iter()
            .position(|d| d.id == query || d.name == query)
            .or_else(|| devices.iter().position(|d| d.name.to_lowercase() == lower))
            .or_else(|| {
                devices
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.name.to_lowercase().contains(&lower))
                    .min_by_key(|(_, d)| d.name.len())
                    .map(|(i, _)| i)
            })
    };
    match index {
        Some(index) => Ok(devices.swap_remove(index)),
        None => {
            let names: Vec<String> = devices
                .iter()
                .map(|d| format!("{:?} (id {})", d.name, d.id))
                .collect();
            Err(anyhow!(
                "no {what} matches {query:?}; valid: \"default\", {}",
                names.join(", ")
            ))
        }
    }
}

/// The lowest and highest of an integer caps field, fixed or a range.
pub(crate) fn int_field(caps: Option<&gst::Caps>, field: &str) -> Option<(i32, i32)> {
    let structure = caps?.structure(0)?;
    if let Ok(value) = structure.get::<i32>(field) {
        return Some((value, value));
    }
    let range = structure.get::<gst::IntRange<i32>>(field).ok()?;
    Some((range.min(), range.max()))
}

/// The channels a device delivers (or takes): its own count, else stereo.
pub(crate) fn channels(device: &gst::Device) -> u32 {
    match int_field(device.caps().as_ref(), "channels") {
        Some((min, max)) if min == max => min.clamp(1, 8) as u32,
        _ => 2,
    }
}

/// The rates a device offers among the usual ones.
fn sample_rates(device: &gst::Device) -> Vec<f64> {
    match int_field(device.caps().as_ref(), "rate") {
        Some((min, max)) if min == max => vec![f64::from(min)],
        Some((min, max)) => [44_100, 48_000, 88_200, 96_000, 192_000]
            .into_iter()
            .filter(|rate| (min..=max).contains(rate))
            .map(f64::from)
            .collect(),
        // PipeWire resamples whatever is asked of it.
        None => vec![48_000.0],
    }
}

pub(crate) fn microphones() -> Result<Vec<Microphone>> {
    Ok(find_all("Audio/Source")?
        .into_iter()
        .map(|found| {
            let properties = found.device.properties();
            let string = |key: &str| {
                properties
                    .as_ref()
                    .and_then(|p| p.get::<String>(key).ok())
                    .unwrap_or_default()
            };
            Microphone {
                model: string("device.product.name"),
                manufacturer: string("device.vendor.name"),
                default: found.default,
                sample_rates: sample_rates(&found.device),
                channels: channels(&found.device),
                id: found.id,
                name: found.name,
            }
        })
        .collect())
}

pub(crate) fn audio_outputs() -> Result<Vec<AudioOutput>> {
    Ok(find_all("Audio/Sink")?
        .into_iter()
        .map(|found| AudioOutput {
            sample_rate: sample_rates(&found.device)
                .into_iter()
                .find(|rate| *rate == 48_000.0)
                .unwrap_or(48_000.0),
            channels: channels(&found.device),
            default: found.default,
            id: found.id,
            name: found.name,
        })
        .collect())
}
