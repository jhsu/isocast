//! The clock, volumes and the machine.

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use anyhow::Result;
use isocast_engine::{Time, Volume};
use isocast_protocol::api::Machine;

/// File systems that are network shares.
const NETWORK_FILE_SYSTEMS: &[&str] = &[
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "9p",
    "ceph",
    "fuse.sshfs",
    "afs",
];

/// `CLOCK_MONOTONIC`, which GStreamer and PipeWire stamp buffers on too.
pub fn host_time() -> Time {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes the monotonic clock into `time`.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    Time::from_nanos(time.tv_sec * 1_000_000_000 + time.tv_nsec)
}

/// The CPU time the calling thread has used.
pub fn thread_cpu_time() -> Option<std::time::Duration> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes the calling thread's CPU time into `time`.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    (status == 0)
        .then(|| std::time::Duration::new(time.tv_sec.max(0) as u64, time.tv_nsec.max(0) as u32))
}

/// The volume holding `path`: space available to an unprivileged process,
/// its file system (from the longest mount point holding it), and whether
/// it is a network share.
pub fn volume(path: &Path) -> Result<Volume> {
    let c_path = CString::new(path.as_os_str().as_bytes())?;
    let mut stats = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is NUL-terminated; statvfs fills `stats` on success.
    if unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: filled in above.
    let stats = unsafe { stats.assume_init() };
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
    let file_system = mounts
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(' ');
            let _device = fields.next()?;
            let mount = fields.next()?.replace("\\040", " ");
            let kind = fields.next()?;
            path.starts_with(&mount)
                .then(|| (mount.len(), kind.to_string()))
        })
        .max_by_key(|(len, _)| *len)
        .map(|(_, kind)| kind)
        .unwrap_or_default();
    Ok(Volume {
        free_bytes: stats.f_bavail * stats.f_frsize,
        network: NETWORK_FILE_SYSTEMS.contains(&file_system.as_str()),
        file_system,
    })
}

/// Write 4 KB at the start of `probe` and fdatasync it, timed.
pub fn write_latency(probe: &Path) -> std::io::Result<std::time::Duration> {
    use std::os::unix::fs::FileExt;
    let started = std::time::Instant::now();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(probe)?;
    file.write_all_at(&[0; 4096], 0)?;
    file.sync_data()?;
    Ok(started.elapsed())
}

/// The CPU, memory, distribution and the hardware video encoders
/// GStreamer can make.
pub fn machine() -> Machine {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let chip = cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name")?.split_once(':'))
        .map(|(_, name)| name.trim().to_string())
        .unwrap_or_else(|| std::env::consts::ARCH.into());
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let memory_kb: f64 = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0.0);
    let release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    let distribution = release
        .lines()
        .find_map(|line| line.strip_prefix("PRETTY_NAME="))
        .map(|name| name.trim_matches('"').to_string())
        .unwrap_or_else(|| "Linux".into());
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    Machine {
        chip,
        memory_gb: (memory_kb / 1024.0 / 1024.0 * 10.0).round() / 10.0,
        cores: std::thread::available_parallelism().map_or(0, |n| n.get() as u32),
        os: format!("{distribution} (Linux {})", kernel.trim()),
        hardware_encoders: hardware_encoders(),
    }
}

fn hardware_encoders() -> Vec<String> {
    if crate::writer::init().is_err() {
        return Vec::new();
    }
    [
        ("nvh264enc", "NVENC H.264"),
        ("nvh265enc", "NVENC HEVC"),
        ("vah264enc", "VA-API H.264"),
        ("vah265enc", "VA-API HEVC"),
    ]
    .into_iter()
    .filter(|(element, _)| gstreamer::ElementFactory::find(element).is_some())
    .map(|(_, name)| name.to_string())
    .collect()
}
