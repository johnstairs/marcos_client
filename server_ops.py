"""Strongly-typed wrappers for MaRCoS server operations.

Each function corresponds to a command understood by ``hardware::run_request``
on the server side and returns a typed result extracted from the reply.

All wrappers accept the common *print_infos* / *assert_errors* flags from
:func:`server_comms.command` and forward them unchanged.  Any extra
``**params`` are forwarded as msgpack request parameters (e.g.
``stream_response=True``).

Backwards-compatible: callers that prefer the raw ``(Reply, StatusDict)``
tuple can keep using :func:`server_comms.command` directly.
"""

from beartype import beartype
from collections.abc import Iterator
from socket import socket as _Socket
from typing import Any, Literal, NamedTuple, TypedDict

from server_comms import (
    CommandResult,
    Reply,
    StatusDict,
    command,
    close_server_pkt,
    construct_packet,
    send_packet,
    streamed_command,
)

__all__ = [
    # Result types
    "BusTimings",
    "NetThroughput",
    "RegStatus",
    "RxData",
    "ServerKind",
    # Operations
    "acq_rlim",
    "are_you_real",
    "close_server",
    "ctrl",
    "direct",
    "fpga_clk",
    "halt_and_reset",
    "mar_mem",
    "read_mem",
    "read_rx",
    "regrd",
    "regstatus",
    "run_seq",
    "run_seq_streamed",
    "server_version",
    "set_gpa_zero_words",
    "test_bus",
    "test_net",
]

# ── Shared result types ──────────────────────────────────────────────

class RxData(TypedDict, total=False):
    """RX sample arrays returned by ``read_rx`` and ``run_seq``."""
    rx0_i: list[int]
    rx0_q: list[int]
    rx1_i: list[int]
    rx1_q: list[int]


class RegStatus(NamedTuple):
    """Register snapshot returned by ``regstatus``."""
    exec: int
    status: int
    status_latch: int
    buf_err: int
    buf_full: int
    buf_empty: int
    rx_locs: int


class BusTimings(NamedTuple):
    """Microsecond timings returned by ``test_bus``."""
    null_us: int
    read_us: int
    write_us: int


class NetThroughput(TypedDict):
    """Arrays returned by ``test_net``."""
    array1: list[float]
    array2: list[float]


# ── Helpers ──────────────────────────────────────────────────────────

def _cmd(
    key: str, value: Any, socket: _Socket,
    print_infos: bool = False, assert_errors: bool = False,
    **params: Any,
) -> CommandResult:
    return command({key: value}, socket, print_infos, assert_errors, **params)


# ── Operations ───────────────────────────────────────────────────────

@beartype
def halt_and_reset(
    socket: _Socket, *, print_infos: bool = False, assert_errors: bool = False,
) -> tuple[bool, StatusDict]:
    """Halt and reset the FSM.  Returns ``True`` if the FSM has halted."""
    reply, status = _cmd("halt_and_reset", 0, socket, print_infos, assert_errors)
    return reply.data["halt_and_reset"], status


@beartype
def set_gpa_zero_words(
    words: list[int], socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Register the per-channel "zero current" direct-write words used
    by ``halt_and_reset`` to park the gradient DACs at midpoint.

    *words* is a list of 32-bit gradient-serialiser frames (one per
    channel) computed client-side via ``grad_board.float2bin`` and
    ``marcompile.col2buf``. Old servers that do not implement this
    command will simply ignore it (status ``-1``); the cancel path
    will then leave the DACs latched at their last sample.
    """
    reply, status = _cmd("set_gpa_zero_words", list(words), socket,
                         print_infos, assert_errors)
    return reply.data.get("set_gpa_zero_words", 0), status


@beartype
def read_mem(
    socket: _Socket, *, print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Read directly from memory (server TODO).  Returns ``0`` on success."""
    reply, status = _cmd("read_mem", 0, socket, print_infos, assert_errors)
    return reply.data["read_mem"], status


@beartype
def fpga_clk(
    words: tuple[int, int, int], socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Configure the FPGA clock.  *words* is a 3-element tuple of uint32 values.
    Returns ``0`` on success or ``-1`` on error."""
    reply, status = _cmd("fpga_clk", list(words), socket, print_infos, assert_errors)
    return reply.data["fpga_clk"], status


@beartype
def ctrl(
    value: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Write to the main control register.  Returns ``0`` on success."""
    reply, status = _cmd("ctrl", value, socket, print_infos, assert_errors)
    return reply.data["ctrl"], status


@beartype
def direct(
    value: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Write directly to a buffer.  Returns ``0`` on success."""
    reply, status = _cmd("direct", value, socket, print_infos, assert_errors)
    return reply.data["direct"], status


@beartype
def regrd(
    index: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Read one hardware register by *index*.  Returns the register value."""
    reply, status = _cmd("regrd", index, socket, print_infos, assert_errors)
    return reply.data["regrd"], status


@beartype
def regstatus(
    socket: _Socket, *, print_infos: bool = False, assert_errors: bool = False,
) -> tuple[RegStatus, StatusDict]:
    """Read all status registers.  Returns a :class:`RegStatus` named tuple."""
    reply, status = _cmd("regstatus", 0, socket, print_infos, assert_errors)
    return RegStatus(*reply.data["regstatus"]), status


@beartype
def mar_mem(
    data: bytes, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Write execution memory.  Returns ``0`` on success or ``-1`` on error."""
    reply, status = _cmd("mar_mem", data, socket, print_infos, assert_errors)
    return reply.data["mar_mem"], status


@beartype
def acq_rlim(
    limit: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[int, StatusDict]:
    """Configure the acquisition retry limit (must be in [1000, 10_000_000]).
    Returns ``0`` on success or ``-1`` on error."""
    reply, status = _cmd("acq_rlim", limit, socket, print_infos, assert_errors)
    return reply.data["acq_rlim"], status


@beartype
def read_rx(
    socket: _Socket, *, print_infos: bool = False, assert_errors: bool = False,
) -> tuple[RxData | int, StatusDict]:
    """Read outstanding RX FIFO data.  Returns an :class:`RxData` dict, or
    ``0`` if there was no data."""
    reply, status = _cmd("read_rx", 0, socket, print_infos, assert_errors)
    return reply.data["read_rx"], status


@beartype
def run_seq(
    bytecode: bytes, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[RxData | int, StatusDict]:
    """Run a compiled sequence.  Returns an :class:`RxData` dict, or ``0``
    if no RX data was received."""
    reply, status = _cmd("run_seq", bytecode, socket, print_infos, assert_errors)
    return reply.data["run_seq"], status


@beartype
def run_seq_streamed(
    bytecode: bytes, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> Iterator[RxData | tuple[RxData | int, StatusDict]]:
    """Run a sequence with RX streaming.

    Yields intermediate :class:`RxData` chunks.  The last yielded value is a
    ``(RxData | int, StatusDict)`` tuple with the final reply."""
    for msg in streamed_command(
        {"run_seq": bytecode}, socket,
        print_infos=print_infos, assert_errors=assert_errors,
        stream_response=True,
    ):
        if isinstance(msg, tuple):
            reply, status = msg
            yield reply.data["run_seq"], status
        else:
            # intermediate chunk: [type, index, {rx0_i: [...], ...}]
            yield msg[2]


@beartype
def test_net(
    data_size: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[NetThroughput, StatusDict]:
    """Test client-server network throughput with *data_size* elements."""
    reply, status = _cmd("test_net", data_size, socket, print_infos, assert_errors)
    return reply.data["test_net"], status


@beartype
def test_bus(
    n_tests: int, socket: _Socket, *,
    print_infos: bool = False, assert_errors: bool = False,
) -> tuple[BusTimings, StatusDict]:
    """Test bus read/write throughput.  Returns microsecond timings."""
    reply, status = _cmd("test_bus", n_tests, socket, print_infos, assert_errors)
    return BusTimings(*reply.data["test_bus"]), status


ServerKind = Literal["hardware", "simulation", "software"]


@beartype
def are_you_real(
    socket: _Socket, *, print_infos: bool = False, assert_errors: bool = False,
) -> tuple[ServerKind, StatusDict]:
    """Check whether the server runs on hardware, simulation, or software."""
    reply, status = _cmd("are_you_real", 0, socket, print_infos, assert_errors)
    return reply.data["are_you_real"], status


@beartype
def close_server(socket: _Socket) -> Reply:
    """Send the close-server packet.  The server will shut down."""
    return send_packet(construct_packet({}, 0, command=close_server_pkt), socket)


@beartype
def server_version(socket: _Socket) -> int:
    """Return the server's protocol version uint.

    Sends a cheap ``are_you_real`` probe and extracts the version from the
    protocol frame (``Reply.version``)."""
    reply, _status = _cmd("are_you_real", 0, socket)
    return reply.version
