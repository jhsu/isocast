//! Displays, from the Wayland compositor's outputs (`wl_output` 4: its
//! connector name, make, model, position and current mode). The portal
//! can't list them; it only lets the user pick one.

use anyhow::{Context, Result, anyhow};
use isocast_protocol::devices::Display;
use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};

#[derive(Default)]
struct Output {
    connector: String,
    make: String,
    model: String,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    refresh_mhz: i32,
}

#[derive(Default)]
struct Outputs(Vec<Output>);

impl Dispatch<wl_registry::WlRegistry, ()> for Outputs {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
            && interface == "wl_output"
        {
            registry.bind::<wl_output::WlOutput, _, _>(name, version.min(4), queue, state.0.len());
            state.0.push(Output::default());
        }
    }
}

impl Dispatch<wl_output::WlOutput, usize> for Outputs {
    fn event(
        state: &mut Self,
        _: &wl_output::WlOutput,
        event: wl_output::Event,
        index: &usize,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let output = &mut state.0[*index];
        match event {
            wl_output::Event::Geometry {
                x, y, make, model, ..
            } => {
                (output.x, output.y) = (x, y);
                output.make = make;
                output.model = model;
            }
            wl_output::Event::Mode {
                flags: WEnum::Value(flags),
                width,
                height,
                refresh,
            } if flags.contains(wl_output::Mode::Current) => {
                output.width = width.max(0) as u32;
                output.height = height.max(0) as u32;
                output.refresh_mhz = refresh;
            }
            wl_output::Event::Name { name } => output.connector = name,
            _ => {}
        }
    }
}

/// The displays, or an error if this isn't a Wayland session.
pub(crate) fn displays() -> Result<Vec<Display>> {
    let connection = Connection::connect_to_env()
        .context("listing displays needs a Wayland session (WAYLAND_DISPLAY)")?;
    let mut queue = connection.new_event_queue();
    let handle = queue.handle();
    connection.display().get_registry(&handle, ());
    let mut outputs = Outputs::default();
    // Once for the globals, once for each output's description.
    queue.roundtrip(&mut outputs)?;
    queue.roundtrip(&mut outputs)?;
    let main = outputs
        .0
        .iter()
        .position(|o| (o.x, o.y) == (0, 0))
        .unwrap_or(0);
    Ok(outputs
        .0
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            let name = format!("{} {}", output.make, output.model)
                .trim()
                .to_string();
            Display {
                built_in: ["eDP", "LVDS", "DSI"]
                    .iter()
                    .any(|prefix| output.connector.starts_with(prefix)),
                name: if name.is_empty() {
                    output.connector.clone()
                } else {
                    name
                },
                id: output.connector,
                main: index == main,
                width: output.width,
                height: output.height,
                refresh_rate: f64::from(output.refresh_mhz) / 1000.0,
            }
        })
        .collect())
}

/// The display `query` names: "main", its connector (like `DP-3`) or exact
/// name, else its name in any case, else the shortest name containing it.
pub(crate) fn find(query: &str) -> Result<Display> {
    let mut displays = displays()?;
    let lower = query.to_lowercase();
    let index = if query == "main" {
        displays.iter().position(|d| d.main)
    } else {
        displays
            .iter()
            .position(|d| d.id == query || d.name == query)
            .or_else(|| displays.iter().position(|d| d.name.to_lowercase() == lower))
            .or_else(|| {
                displays
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.name.to_lowercase().contains(&lower))
                    .min_by_key(|(_, d)| d.name.len())
                    .map(|(i, _)| i)
            })
    };
    match index {
        Some(index) => Ok(displays.swap_remove(index)),
        None => {
            let names: Vec<String> = displays
                .iter()
                .map(|d| format!("{:?} (id {})", d.name, d.id))
                .collect();
            Err(anyhow!(
                "no display matches {query:?}; valid: \"main\", {}",
                names.join(", ")
            ))
        }
    }
}
