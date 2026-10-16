//! Vulkan images whose memory is exported as file descriptors
//! (`VK_KHR_external_memory_fd`, opaque fds), handed between processes over
//! a Unix socket (`SCM_RIGHTS`).
//!
//! The daemon listens on an abstract socket of its own
//! (`isocastd.<pid>.surfaces`: nothing on disk, gone when the process
//! exits) and only answers its own user. A client sends a token; the reply
//! carries each surface's size and its GPU's ids, with one descriptor per
//! surface, which the client imports into its own Vulkan device. Opaque fds
//! only open on the same GPU and driver, so the ids are checked first.
//!
//! Unlike an IOSurface, a surface's memory belongs to a GPU device, so the
//! daemon makes surfaces with [`Surface::with_texture`], not
//! [`Surface::rgba`]. Between hand-overs the image is kept in the layout
//! for sampling (`TextureUses::RESOURCE`): the daemon moves it there after
//! drawing, and clients only sample it.

use std::io::{Read, Write};
use std::mem::{size_of, zeroed};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::{MAX_SLOTS, Token};

/// How the daemon says it hands surfaces over (`SharedFrameStream::transport`).
pub const TRANSPORT: &str = "vulkan-fd";

/// How long the daemon waits for a client's token, or to send its reply.
const SERVE_TIMEOUT: Duration = Duration::from_secs(1);

/// The reply's first bytes: "ISOS", then whether the token was taken.
const MAGIC: u32 = u32::from_le_bytes(*b"ISOS");
const GRANTED: u32 = 0;
const REFUSED: u32 = 1;
/// Magic, status and count.
const HEADER: usize = 12;
/// Width, height, allocation size, device and driver ids.
const ENTRY: usize = 4 + 4 + 8 + 16 + 16;

/// An 8-bit RGBA picture in GPU memory that several processes can use at
/// once. Cloning shares the same surface.
#[derive(Clone)]
pub struct Surface(Arc<Inner>);

struct Inner {
    id: u32,
    width: u32,
    height: u32,
    /// The size of the memory allocation, which an import must repeat.
    size: u64,
    /// The exported memory; each import consumes a copy of it.
    fd: OwnedFd,
    /// `VkPhysicalDeviceIDProperties` of the GPU the memory is on.
    device_uuid: [u8; 16],
    driver_uuid: [u8; 16],
}

impl Surface {
    /// Not on Linux: a surface's memory belongs to a GPU device; see
    /// [`Surface::with_texture`].
    pub fn rgba(_width: u32, _height: u32) -> Result<Self> {
        bail!("on Linux a shared surface is made on a GPU device (Surface::with_texture)")
    }

    pub fn width(&self) -> u32 {
        self.0.width
    }

    pub fn height(&self) -> u32 {
        self.0.height
    }

    /// A number for logs, unique in this process (not a way to share it).
    pub fn id(&self) -> u32 {
        self.0.id
    }

    /// A new `width`×`height` surface of 8-bit RGBA on `device` (a Vulkan
    /// device), its contents undefined, and the texture of `device` that
    /// draws into it. Ask for `TEXTURE_BINDING` too, so the picture can be
    /// moved to the layout clients sample it in after each frame.
    #[cfg(feature = "wgpu")]
    pub fn with_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        usage: wgpu::TextureUsages,
    ) -> Result<(Self, wgpu::Texture)> {
        vulkan::create(device, width, height, usage)
    }

    /// The surface as an `Rgba8Unorm` texture of `device` (a Vulkan device
    /// on the same GPU and driver as the daemon's) with `usage`, in place:
    /// the GPU reads the surface's own memory. For Slint's
    /// `Image::try_from(wgpu::Texture)` ask for `TEXTURE_BINDING |
    /// RENDER_ATTACHMENT`.
    #[cfg(feature = "wgpu")]
    pub fn texture(
        &self,
        device: &wgpu::Device,
        usage: wgpu::TextureUsages,
    ) -> Result<wgpu::Texture> {
        vulkan::import(&self.0, device, usage)
    }
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Surface({} {}×{})",
            self.id(),
            self.width(),
            self.height()
        )
    }
}

fn next_id() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The host clock now, in nanoseconds: `CLOCK_MONOTONIC`, the clock frame
/// times are on.
pub fn host_now() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes the monotonic clock into `time`.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

/// The daemon's side: a socket that hands each offered ring of surfaces to
/// the client holding its token, once.
pub struct Exporter {
    service: String,
    offers: Arc<Offers>,
}

/// Rings offered and not fetched yet, by token.
type Offers = Mutex<std::collections::HashMap<Token, Vec<Surface>>>;

impl Exporter {
    /// This process's exporter, its socket opened on first use.
    pub fn shared() -> Result<&'static Exporter> {
        static SHARED: OnceLock<Result<Exporter, String>> = OnceLock::new();
        SHARED
            .get_or_init(|| Exporter::start().map_err(|err| format!("{err:#}")))
            .as_ref()
            .map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn start() -> Result<Exporter> {
        let service = format!("isocastd.{}.surfaces", std::process::id());
        let address = SocketAddr::from_abstract_name(service.as_bytes())?;
        let listener = UnixListener::bind_addr(&address)
            .with_context(|| format!("could not open the surface socket @{service}"))?;
        let offers = Arc::new(Offers::default());
        let served = offers.clone();
        thread::Builder::new()
            .name("shared surfaces".into())
            .spawn(move || serve(&listener, &served))
            .context("could not start the shared surface service")?;
        Ok(Exporter { service, offers })
    }

    /// The abstract socket's name, which clients connect to.
    pub fn service(&self) -> &str {
        &self.service
    }

    /// Offer `surfaces` (at most [`MAX_SLOTS`]) to whoever brings the token
    /// returned, once.
    pub fn offer(&self, surfaces: Vec<Surface>) -> Token {
        assert!(surfaces.len() <= MAX_SLOTS, "too many surfaces in a ring");
        let token = Token::random();
        self.offers.lock().unwrap().insert(token, surfaces);
        token
    }

    /// Take back an offer nobody fetched.
    pub fn withdraw(&self, token: Token) {
        self.offers.lock().unwrap().remove(&token);
    }
}

/// Answer clients for as long as the process runs, one at a time (each
/// exchange is a few hundred bytes).
fn serve(listener: &UnixListener, offers: &Offers) {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        // Another user's process may connect to an abstract socket; only
        // our own user gets an answer.
        if peer_uid(&stream) != Some(unsafe { libc::getuid() }) {
            continue;
        }
        let _ = stream.set_read_timeout(Some(SERVE_TIMEOUT));
        let _ = stream.set_write_timeout(Some(SERVE_TIMEOUT));
        let mut token = [0u8; 16];
        if stream.read_exact(&mut token).is_err() {
            continue;
        }
        let surfaces = offers.lock().unwrap().remove(&Token(token));
        let _ = reply(&stream, surfaces.as_deref());
    }
}

fn peer_uid(stream: &UnixStream) -> Option<libc::uid_t> {
    let mut cred: libc::ucred = unsafe { zeroed() };
    let mut len = size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED fills a ucred of `len` bytes.
    let status = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&raw mut cred).cast(),
            &mut len,
        )
    };
    (status == 0).then_some(cred.uid)
}

/// Send `surfaces` (or a refusal) to the client on `stream`.
fn reply(stream: &UnixStream, surfaces: Option<&[Surface]>) -> std::io::Result<()> {
    let surfaces = surfaces.unwrap_or_default();
    let status = if surfaces.is_empty() {
        REFUSED
    } else {
        GRANTED
    };
    let mut message = Vec::with_capacity(HEADER + surfaces.len() * ENTRY);
    message.extend_from_slice(&MAGIC.to_le_bytes());
    message.extend_from_slice(&status.to_le_bytes());
    message.extend_from_slice(&(surfaces.len() as u32).to_le_bytes());
    for surface in surfaces {
        let inner = &surface.0;
        message.extend_from_slice(&inner.width.to_le_bytes());
        message.extend_from_slice(&inner.height.to_le_bytes());
        message.extend_from_slice(&inner.size.to_le_bytes());
        message.extend_from_slice(&inner.device_uuid);
        message.extend_from_slice(&inner.driver_uuid);
    }
    let fds: Vec<RawFd> = surfaces.iter().map(|s| s.0.fd.as_raw_fd()).collect();
    send_with_fds(stream, &message, &fds)
}

/// The client's side: fetch the surfaces offered under `token` from the
/// daemon's socket `service`, waiting up to `timeout`.
pub fn fetch(service: &str, token: Token, timeout: Duration) -> Result<Vec<Surface>> {
    let address = SocketAddr::from_abstract_name(service.as_bytes())?;
    let mut stream = UnixStream::connect_addr(&address).with_context(|| {
        format!(
            "could not reach isocastd's surface socket @{service}; the daemon may have \
             stopped, or run in another network namespace (a sandbox)"
        )
    })?;
    let timeout = Some(timeout.max(Duration::from_millis(1)));
    stream.set_read_timeout(timeout)?;
    stream.set_write_timeout(timeout)?;
    stream
        .write_all(&token.0)
        .context("could not send a request to isocastd's surface socket")?;
    let mut message = vec![0u8; HEADER + MAX_SLOTS * ENTRY];
    let (mut read, mut fds) =
        recv_with_fds(&stream, &mut message).context("could not receive surfaces from isocastd")?;
    // The rest of a reply that came in pieces (the fds come with the first).
    while read < HEADER {
        let more = stream.read(&mut message[read..])?;
        if more == 0 {
            bail!("isocastd's surface socket closed before replying");
        }
        read += more;
    }
    let u32_at = |at: usize| u32::from_le_bytes(message[at..at + 4].try_into().unwrap());
    if u32_at(0) != MAGIC {
        bail!("an unexpected reply from isocastd's surface socket");
    }
    if u32_at(4) == REFUSED {
        bail!("isocastd refused the token: it was used already or the ring was replaced");
    }
    let count = u32_at(8) as usize;
    if count > MAX_SLOTS || count != fds.len() {
        bail!(
            "isocastd's surface socket sent {} descriptors for {count} surfaces",
            fds.len()
        );
    }
    let length = HEADER + count * ENTRY;
    while read < length {
        let more = stream.read(&mut message[read..length])?;
        if more == 0 {
            bail!("isocastd's surface socket closed during its reply");
        }
        read += more;
    }
    let mut surfaces = Vec::with_capacity(count);
    for (index, fd) in fds.drain(..).enumerate() {
        let entry = &message[HEADER + index * ENTRY..HEADER + (index + 1) * ENTRY];
        surfaces.push(Surface(Arc::new(Inner {
            id: next_id(),
            width: u32::from_le_bytes(entry[0..4].try_into().unwrap()),
            height: u32::from_le_bytes(entry[4..8].try_into().unwrap()),
            size: u64::from_le_bytes(entry[8..16].try_into().unwrap()),
            fd,
            device_uuid: entry[16..32].try_into().unwrap(),
            driver_uuid: entry[32..48].try_into().unwrap(),
        })));
    }
    Ok(surfaces)
}

/// Room for [`MAX_SLOTS`] descriptors in a control message.
fn control_space() -> usize {
    // SAFETY: arithmetic only.
    unsafe { libc::CMSG_SPACE((MAX_SLOTS * size_of::<RawFd>()) as u32) as usize }
}

/// Send `data` with copies of `fds` attached, in one message.
fn send_with_fds(stream: &UnixStream, data: &[u8], fds: &[RawFd]) -> std::io::Result<()> {
    let mut iov = libc::iovec {
        iov_base: data.as_ptr() as *mut _,
        iov_len: data.len(),
    };
    let mut control = vec![0u8; control_space()];
    let mut header: libc::msghdr = unsafe { zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    if !fds.is_empty() {
        let length = size_of_val(fds);
        // SAFETY: `control` has room for MAX_SLOTS descriptors and `fds`
        // has at most that many (Exporter::offer).
        unsafe {
            header.msg_control = control.as_mut_ptr().cast();
            header.msg_controllen = libc::CMSG_SPACE(length as u32) as _;
            let cmsg = libc::CMSG_FIRSTHDR(&header);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(length as u32) as _;
            std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(cmsg).cast(), fds.len());
        }
    }
    // SAFETY: `header` points at live buffers for the call's duration.
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &header, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Small messages go whole on a Unix socket; send any rest plainly.
    let mut stream = stream;
    stream.write_all(&data[sent as usize..])
}

/// Receive into `buffer` and take the descriptors attached (close-on-exec).
fn recv_with_fds(stream: &UnixStream, buffer: &mut [u8]) -> std::io::Result<(usize, Vec<OwnedFd>)> {
    let mut iov = libc::iovec {
        iov_base: buffer.as_mut_ptr().cast(),
        iov_len: buffer.len(),
    };
    let mut control = vec![0u8; control_space()];
    let mut header: libc::msghdr = unsafe { zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.as_mut_ptr().cast();
    header.msg_controllen = control.len() as _;
    // SAFETY: `header` points at live buffers for the call's duration.
    let read = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut header, libc::MSG_CMSG_CLOEXEC) };
    if read < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut fds = Vec::new();
    // SAFETY: walks the control messages the kernel wrote into `control`.
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&header);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                let bytes = (*cmsg).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                for index in 0..bytes / size_of::<RawFd>() {
                    fds.push(OwnedFd::from_raw_fd(data.add(index).read_unaligned()));
                }
            }
            cmsg = libc::CMSG_NXTHDR(&header, cmsg);
        }
    }
    if header.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(std::io::Error::other("too many descriptors in the reply"));
    }
    Ok((read as usize, fds))
}

/// Making, exporting and importing the images, through wgpu-hal's Vulkan
/// device.
#[cfg(feature = "wgpu")]
mod vulkan {
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
    use std::sync::Arc;

    use anyhow::{Context, Result, bail};
    use ash::vk;
    use wgpu::hal;

    use super::{Inner, Surface, next_id};

    const HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

    /// Every surface is made the same way, whatever each side's wgpu usage:
    /// an import must describe the image exactly as the export did.
    fn image_info<'a>(
        width: u32,
        height: u32,
        external: &'a mut vk::ExternalMemoryImageCreateInfo<'_>,
    ) -> vk::ImageCreateInfo<'a> {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(external)
    }

    /// The usages wgpu-hal tracks for `usage`.
    fn uses(usage: wgpu::TextureUsages) -> Result<wgpu::TextureUses> {
        let mut uses = wgpu::TextureUses::empty();
        for (asked, use_) in [
            (wgpu::TextureUsages::COPY_SRC, wgpu::TextureUses::COPY_SRC),
            (wgpu::TextureUsages::COPY_DST, wgpu::TextureUses::COPY_DST),
            (
                wgpu::TextureUsages::TEXTURE_BINDING,
                wgpu::TextureUses::RESOURCE,
            ),
            (
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                wgpu::TextureUses::COLOR_TARGET,
            ),
        ] {
            if usage.contains(asked) {
                uses |= use_;
            }
        }
        let others = usage
            - (wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT);
        if !others.is_empty() {
            bail!("a shared surface can't be used as {others:?}");
        }
        Ok(uses)
    }

    /// The device's and driver's UUIDs: an opaque fd only opens where both
    /// match.
    fn ids(device: &hal::vulkan::Device) -> Result<([u8; 16], [u8; 16])> {
        let instance = device.shared_instance();
        if instance.instance_api_version() < vk::API_VERSION_1_1 {
            bail!("sharing GPU surfaces needs Vulkan 1.1");
        }
        let mut id = vk::PhysicalDeviceIDProperties::default();
        {
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            // SAFETY: a valid physical device of this instance.
            unsafe {
                instance
                    .raw_instance()
                    .get_physical_device_properties2(device.raw_physical_device(), &mut properties)
            };
        }
        Ok((id.device_uuid, id.driver_uuid))
    }

    fn fd_functions(device: &hal::vulkan::Device) -> Result<ash::khr::external_memory_fd::Device> {
        if !device
            .enabled_device_extensions()
            .contains(&ash::khr::external_memory_fd::NAME)
        {
            bail!("the GPU's Vulkan driver can't share memory (no VK_KHR_external_memory_fd)");
        }
        Ok(ash::khr::external_memory_fd::Device::new(
            device.shared_instance().raw_instance(),
            device.raw_device(),
        ))
    }

    /// A device-local memory type among `bits`.
    fn memory_type(device: &hal::vulkan::Device, bits: u32) -> Result<u32> {
        // SAFETY: a valid physical device of this instance.
        let properties = unsafe {
            device
                .shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(device.raw_physical_device())
        };
        properties
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .find(|(index, kind)| {
                bits & (1 << index) != 0
                    && kind
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .map(|(index, _)| index as u32)
            .context("the GPU has no memory a shared surface can be in")
    }

    /// An image made for sharing, with its memory requirements.
    fn create_image(
        device: &hal::vulkan::Device,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::MemoryRequirements)> {
        if width == 0 || height == 0 {
            bail!("a surface can't be {width}×{height}");
        }
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(HANDLE);
        let raw = device.raw_device();
        // SAFETY: a valid create info; the image is destroyed or handed to
        // wgpu-hal by the callers.
        let image = unsafe { raw.create_image(&image_info(width, height, &mut external), None) }
            .with_context(|| format!("could not create a {width}×{height} shared image"))?;
        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        Ok((image, requirements))
    }

    /// Allocate `memory` for `image` (dedicated, as external images want)
    /// and bind it; the image is destroyed on failure.
    fn bind_memory(
        device: &hal::vulkan::Device,
        image: vk::Image,
        info: vk::MemoryAllocateInfo<'_>,
    ) -> Result<vk::DeviceMemory> {
        let raw = device.raw_device();
        // SAFETY: `info` describes memory for `image`.
        let memory = match unsafe { raw.allocate_memory(&info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { raw.destroy_image(image, None) };
                return Err(err).context("could not allocate a shared surface's memory");
            }
        };
        if let Err(err) = unsafe { raw.bind_image_memory(image, memory, 0) } {
            unsafe {
                raw.free_memory(memory, None);
                raw.destroy_image(image, None);
            }
            return Err(err).context("could not bind a shared surface's memory");
        }
        Ok(memory)
    }

    /// Hand `image` and `memory` to wgpu as a texture of `device`, wgpu-hal
    /// owning both, in `state`.
    fn wrap(
        hal_device: &hal::vulkan::Device,
        (image, memory): (vk::Image, vk::DeviceMemory),
        (width, height): (u32, u32),
        (usage, uses): (wgpu::TextureUsages, wgpu::TextureUses),
    ) -> (hal::vulkan::Texture, wgpu::TextureDescriptor<'static>) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let descriptor = hal::TextureDescriptor {
            label: Some("shared surface"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: uses,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        // SAFETY: `image` is bound to `memory`, made as `descriptor` says;
        // wgpu-hal destroys both with the texture.
        let texture = unsafe {
            hal_device.texture_from_raw(
                image,
                &descriptor,
                None,
                hal::vulkan::TextureMemory::Dedicated(memory),
            )
        };
        (
            texture,
            wgpu::TextureDescriptor {
                label: Some("shared surface"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            },
        )
    }

    fn hal_device(
        device: &wgpu::Device,
    ) -> Result<impl std::ops::Deref<Target = hal::vulkan::Device>> {
        // SAFETY: only used to make and wrap resources of this device.
        unsafe { device.as_hal::<hal::api::Vulkan>() }.context("the GPU is not a Vulkan device")
    }

    pub(super) fn create(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        usage: wgpu::TextureUsages,
    ) -> Result<(Surface, wgpu::Texture)> {
        let uses = uses(usage)?;
        let (inner, hal_texture, descriptor) = {
            let hal_device = hal_device(device)?;
            let fds = fd_functions(&hal_device)?;
            let (device_uuid, driver_uuid) = ids(&hal_device)?;
            let (image, requirements) = create_image(&hal_device, width, height)?;
            let memory_type = match memory_type(&hal_device, requirements.memory_type_bits) {
                Ok(index) => index,
                Err(err) => {
                    unsafe { hal_device.raw_device().destroy_image(image, None) };
                    return Err(err);
                }
            };
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(HANDLE);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            let memory = bind_memory(&hal_device, image, info)?;
            let get = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(HANDLE);
            // SAFETY: `memory` was allocated exportable as opaque fds.
            let fd = match unsafe { fds.get_memory_fd(&get) } {
                Ok(fd) => unsafe { OwnedFd::from_raw_fd(fd) },
                Err(err) => {
                    unsafe {
                        hal_device.raw_device().destroy_image(image, None);
                        hal_device.raw_device().free_memory(memory, None);
                    }
                    return Err(err).context("could not export a shared surface's memory");
                }
            };
            let (hal_texture, descriptor) =
                wrap(&hal_device, (image, memory), (width, height), (usage, uses));
            let inner = Inner {
                id: next_id(),
                width,
                height,
                size: requirements.size,
                fd,
                device_uuid,
                driver_uuid,
            };
            (inner, hal_texture, descriptor)
        };
        // SAFETY: a texture of this device, never drawn into yet.
        let texture = unsafe {
            device.create_texture_from_hal::<hal::api::Vulkan>(
                hal_texture,
                &descriptor,
                wgpu::TextureUses::UNINITIALIZED,
            )
        };
        Ok((Surface(Arc::new(inner)), texture))
    }

    pub(super) fn import(
        surface: &Inner,
        device: &wgpu::Device,
        usage: wgpu::TextureUsages,
    ) -> Result<wgpu::Texture> {
        let uses = uses(usage)?;
        let (hal_texture, descriptor) = {
            let hal_device = hal_device(device)?;
            fd_functions(&hal_device)?;
            if ids(&hal_device)? != (surface.device_uuid, surface.driver_uuid) {
                bail!(
                    "isocastd draws on another GPU or driver than this window; shared surfaces \
                     only open on the same one"
                );
            }
            let (image, requirements) = create_image(&hal_device, surface.width, surface.height)?;
            let memory_type = match memory_type(&hal_device, requirements.memory_type_bits) {
                Ok(index) if requirements.size <= surface.size => index,
                result => {
                    unsafe { hal_device.raw_device().destroy_image(image, None) };
                    result?;
                    bail!("a shared surface's memory is smaller than its image needs");
                }
            };
            // Each import consumes a descriptor of its own.
            let fd = surface
                .fd
                .try_clone()
                .context("could not copy a shared surface's descriptor")?;
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(HANDLE)
                .fd(fd.as_raw_fd());
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(surface.size)
                .memory_type_index(memory_type)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            let memory = bind_memory(&hal_device, image, info)?;
            // Vulkan owns the descriptor once the allocation succeeded.
            let _ = fd.into_raw_fd();
            wrap(
                &hal_device,
                (image, memory),
                (surface.width, surface.height),
                (usage, uses),
            )
        };
        // The daemon leaves each finished picture ready for sampling.
        let state = if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            wgpu::TextureUses::RESOURCE
        } else {
            wgpu::TextureUses::COLOR_TARGET
        };
        // SAFETY: a texture of this device over the surface's memory.
        Ok(unsafe {
            device.create_texture_from_hal::<hal::api::Vulkan>(hal_texture, &descriptor, state)
        })
    }
}
