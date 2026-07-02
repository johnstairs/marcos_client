#!/usr/bin/env python3
#
# To run a single test, use e.g.:
# python -m unittest test_server.ServerTest.test_bad_packet

import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
from server_comms import *
import server_ops as ops
import grad_board as gb

class ServerTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    def setUp(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port))
        self.packet_idx = 0

    def tearDown(self):
        self.s.close()

    def test_version(self):
        full_test = True  # test a range of different versions; otherwise just the current one
        debug_replies = False
        def diff_equal(client_ver):
            return {'errors': ['not all client commands were understood']}

        def diff_info(client_ver):
            return {'infos': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' differs slightly from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_warning(client_ver):
            return {'warnings': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_error(client_ver):
            return {'errors': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' significantly different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug),
                               'not all client commands were understood']}

        if full_test:
            versions = [ (1,0,1), (1,0,7), (1,3,100), (1,3,255), (2,5,7), (255,255,255) ]
            expected_outcomes = [diff_info, diff_equal, diff_warning, diff_warning, diff_error, diff_error]
        else:
            versions = [ (1,0,1) ]
            expected_outcomes = [diff_info]


        for v, ee in zip(versions, expected_outcomes):
            # send an unknown command to make sure system handles it gracefully
            packet = construct_packet({'asdfasdf':1}, self.packet_idx, version=v)
            reply = send_packet(packet, self.s)
            expected_reply = (reply_pkt, 1, 0, version_full, {'UNKNOWN1': -1}, ee(v))
            if debug_replies:
                print("Reply         : ", reply)
                print("Expected reply: ", expected_reply)
                if reply == expected_reply:
                    print("Equal")
                else:
                    print("Not equal! Debugging...")
                    st()
            self.assertEqual(reply, expected_reply)

    def test_idle(self):
        """ Make sure the server state is (or becomes) idle, all the RX and TX buffers are empty, etc."""
        real, _ = ops.are_you_real(self.s)
        if real == "hardware" or real == "simulation":
            buf_empties = 0xffffff
        elif real == "software":
            buf_empties = 0

        def check_status():
            """Check status of the firmware, returning True if idle,
            and always returning the last-read ADC value"""
            regs, _ = ops.regstatus(self.s)

            # status fields
            fhdo_busy = 0x20000
            ocra1_busy = 0x10000
            fhdo_adc = 0xffff

            # status latch fields
            fhdo_err = 0x4
            ocra1_err = 0x2
            ocra1_data_lost = 0x1

            if (regs.status_latch & fhdo_err) or (regs.status_latch & ocra1_err) or (regs.status_latch & ocra1_data_lost):
                warnings.warn("Gradient error occurred during test_idle! Might have been caused by another of the tests.")

            adc_value = regs.status & fhdo_adc
            idle = True
            if (regs.status & fhdo_busy) or (regs.status & ocra1_busy):
                idle = False

            return idle, adc_value

        for k in range(1000):
            idle, adc_value = check_status()
            if idle:
                break

        regs, status = ops.regstatus(self.s)
        self.assertEqual(regs, ops.RegStatus(0, adc_value, 0, 0, 0, buf_empties, 0))
        self.assertEqual(status, {})

    def test_bad_packet(self):
        # Manually build a malformed packet (list instead of dict for data)
        # to test the server's error handling — bypasses construct_packet's
        # type checking intentionally.
        packet = Packet(request_pkt, 0, 0, version_full, [1, 2, 3])
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         (reply_pkt, 1, 0, version_full,
                          {},
                          {'errors': ['no commands present or incorrectly formatted request']}))

    def test_bus(self):
        print_speeds = False
        real, _ = ops.are_you_real(self.s)
        if real == "hardware":
            deltas = (2, 2)
            times = (131.0, 158.5) # bus read, bus write on hardware
            loops = 1000000
        elif real == "simulation":
            deltas = (5000, 3500)
            times = (10430, 5000)
            loops = 1000
        elif real == "software":
            deltas = (5, 5)
            times = (7, 7)
            loops = 10000000

        timings, _ = ops.test_bus(loops, self.s)

        loops_norm = loops/1e6
        if print_speeds:
            print(f"{real} data: null_t: {timings.null_us/loops:.2f}, read_t: {timings.read_us/loops:.2f}, write_t: {timings.write_us/loops:.2f} us / cycle")
        if real == "hardware":
            self.assertAlmostEqual(timings.read_us/1e3, times[0] * loops_norm, delta = deltas[0] * loops_norm) # 1 read takes ~141.9 ns on average
            self.assertAlmostEqual(timings.write_us/1e3, times[1] * loops_norm, delta = deltas[1] * loops_norm) # 1 write takes ~157.9 ns on average
        elif real == "simulation":
            # Might not be true if you're on a slow computer, but should be fine for most post-2015 PCs
            self.assertLess(timings.read_us/loops, 100.0)
            self.assertLess(timings.write_us/loops, 100.0)

    @unittest.skip("marga devel")
    def test_net(self):
        real, _ = ops.are_you_real(self.s)
        if real == "hardware":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        elif real == "simulation":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        elif real == "software":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        result, _ = ops.test_net(10, self.s)
        # VN: continue here

    def test_fpga_clk(self):
        result, status = ops.fpga_clk((0xdf0d, 0x03f03f30, 0x00100700), self.s)
        self.assertEqual(result, 0)
        self.assertEqual(status, {})

    def test_fpga_clk_partial(self):
        # Send only 2 words instead of 3 — must use raw API since ops.fpga_clk enforces a 3-tuple
        packet = construct_packet({'fpga_clk': [0xdf0d,  0x03f03f30]})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         (reply_pkt, 1, 0, version_full,
                          {'fpga_clk': -1},
                          {'errors': ["you only provided some FPGA clock control words; check you're providing all 3"]})
        )

    @unittest.skip("marga devel")
    def test_several_okay(self):
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 10,
                                   'rx_div': 250,
                                   'tx_size': 32767,
                                   'raw_tx_data': b"0000000000000000"*4096,
                                   'grad_div': (303, 32),
                                   'grad_ser': 1,
                                   'grad_mem': b"0000"*8192,
                                   'acq_rlim':10000,
                                   })
        reply = send_packet(packet, self.s)

        self.assertEqual(reply,
                         (reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': 0, 'rx_div': 0,
                           'tx_size': 0, 'raw_tx_data': 0, 'grad_div': 0, 'grad_ser': 0,
                           'grad_mem': 0, 'acq_rlim': 0},
                          {'infos': [
                              'tx data bytes copied: 65536',
                              'gradient mem data bytes copied: 32768']})
        )

    @unittest.skip("marga devel")
    def test_several_some_bad(self):
        # first, send a normal packet to ensure everything's in a known state
        packetp = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                    'tx_div': 10, # 81.38ns sampling for 122.88 clock freq, 80ns for 125
                                    'rx_div': 250,
                                    'raw_tx_data': b"0000000000000000"*4096
        })
        send_packet(packetp, self.s)

        # Now, try sending with some issues
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 100000,
                                   'rx_div': 32767,
                                   'tx_size': 65535,
                                   'raw_tx_data': b"0123456789abcdef"*4097,
                                   'grad_div': (1024, 0),
                                   'grad_ser': 16,
                                   'grad_mem': b"0000"*8193,
                                   'acq_rlim': 10,
                                   })

        reply = send_packet(packet, self.s)

        self.assertEqual(reply,
                         (reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': -1, 'rx_div': -1, 'tx_size': -1, 'raw_tx_data': -1, 'grad_div': -1, 'grad_ser': -1, 'grad_mem': -1, 'acq_rlim': -1},
                          {'errors': ['TX divider outside the range [1, 10000]; check your settings',
                                      'RX divider outside the range [25, 8192]; check your settings',
                                      'TX size outside the range [1, 32767]; check your settings',
                                      'too much raw TX data',
                                      'grad SPI clock divider outside the range [1, 63]; check your settings',
                                      'serialiser enables outside the range [0, 0xf], check your settings',
                                      'too much grad mem data: 32772 bytes > 32768',
                                      'acquisition retry limit outside the range [1000, 10,000,000]; check your settings'
                                      ]})
                          )

    @unittest.skipUnless(grad_board == "gpa-fhdo", "requires GPA-FHDO board")
    def test_grad_adc(self):
        print_adc_reads = False
        # initialise SPI
        spi_div = 40
        upd = False # update on MSB writes
        ops.direct(0x00000000 | (2 << 0) | (spi_div << 2) | (0 << 8) | (upd << 9), self.s)

        # ADC defaults, same as in grad_board.GPAFHDO.init_hw().
        # Single source of truth lives on the board class so the two
        # sequences cannot drift apart.
        init_words = gb.GPAFHDO._INIT_WORDS

        real, _ = ops.are_you_real(self.s)
        if real in ['simulation', 'software']:
            expected = [ 0 ] * ( len(init_words) - 1 )
        else:
            expected = [ 0xffff ] + [0x0600] * ( len(init_words) - 2)

        readback = []

        for iw in init_words:
            # direct commands to grad board; send MSBs then LSBs
            ops.direct(0x02000000 | (iw >> 16), self.s)
            ops.direct(0x01000000 | (iw & 0xffff), self.s)

            # read ADC each time

            # status reg = 5, ADC word is lower 16 bits
            adc_read, _ = ops.regrd(5, self.s)
            if print_adc_reads and adc_read != 0:
                print("ADC read: ", adc_read)
            time.sleep(0.01)
            readback.append( adc_read & 0xffff )
            # if readback != r:
            #     warnings.warn( "ADC data expected: 0x{:0x}, observed 0x{:0x}".format(w, readback) )

        self.assertEqual(expected, readback[1:]) # ignore 1st word, since it depends on the history of ADC transfers

    def test_leds(self):
        # This test is mainly for the simulator, but will alter hardware LEDs too
        for k in range(256):
            result, status = ops.direct(0x0f000000 + int((k & 0xff) << 8), self.s)
            self.assertEqual(result, 0)
            self.assertEqual(status, {})

        result, status = ops.direct(0x0f00a500, self.s) # leds: a5
        self.assertEqual(result, 0)
        self.assertEqual(status, {})

        result, status = ops.direct(0x0f002400, self.s) # leds: 24
        self.assertEqual(result, 0)
        self.assertEqual(status, {})

        # kill some time for the LEDs to change in simulation
        for k in range(2):
            ops.regstatus(self.s)

    def test_mar_mem(self):
        mar_mem_bytes = 4 * 65536 # full memory
        # mar_mem_bytes = 4 * 2 # several writes for testing

        # everything should be fine
        raw_data = bytearray(mar_mem_bytes)
        for m in range(mar_mem_bytes):
            raw_data[m] = m & 0xff
        result, status = ops.mar_mem(bytes(raw_data), self.s)
        self.assertEqual(result, 0)
        self.assertEqual(status,
                         {'infos': ['mar mem data bytes copied: {:d}'.format(mar_mem_bytes)] })

        # a bit too much data
        raw_data = bytearray(mar_mem_bytes + 1)
        for m in range(mar_mem_bytes):
            raw_data[m] = m & 0xff
        with self.assertWarns(RuntimeWarning):
            result, status = ops.mar_mem(bytes(raw_data), self.s)
        self.assertEqual(result, -1)
        self.assertEqual(status,
                         {'errors': ['too much mar mem data: {:d} bytes > {:d} -- streaming not yet implemented'.format(mar_mem_bytes + 1, mar_mem_bytes)] })

    def test_set_gpa_zero_words(self):
        """Server accepts a valid 4-word zero-words registration."""
        # The server doesn't interpret the bits; both supported boards
        # (OCRA1, GPA-FHDO) just expose 4 channels. Use the GPA-FHDO
        # production vector so the round-trip exercises the same shape
        # Experiment.__init__ sends in real use.
        words = list(gb.GPAFHDO._GRAD_ZERO_WORDS)
        result, status = ops.set_gpa_zero_words(words, self.s)
        self.assertEqual(result, 0)  # c_ok
        self.assertEqual(status, {})

    def test_set_gpa_zero_words_wrong_count(self):
        """Server rejects a wrong-length word list and preserves the
        previously-registered vector (process-lifetime state)."""
        baseline = [0xaaaaaaaa, 0xbbbbbbbb, 0xcccccccc, 0xdddddddd]
        result, _ = ops.set_gpa_zero_words(baseline, self.s)
        self.assertEqual(result, 0)

        # 3 words instead of 4: server should respond with c_warn (-2),
        # surface a warning in the status map, and leave the baseline
        # intact.
        with self.assertWarns(MarServerWarning):
            result, status = ops.set_gpa_zero_words([0, 0, 0], self.s)
        self.assertEqual(result, -2)  # c_warn
        self.assertIn('warnings', status)
        self.assertEqual(len(status['warnings']), 1)
        msg = status['warnings'][0]
        self.assertIn('received 3 words', msg)
        self.assertIn('expected 4', msg)
        self.assertNotIn('errors', status)

        # Re-registering with a valid count succeeds again. We can't
        # introspect the server-side vector directly, but a c_ok reply
        # combined with the no-error halt_and_reset in
        # ``test_halt_and_reset_uses_registered_zero_words`` covers the
        # "still functional" assertion end-to-end.
        result, _ = ops.set_gpa_zero_words(baseline, self.s)
        self.assertEqual(result, 0)

    def test_halt_and_reset_uses_registered_zero_words(self):
        """End-to-end check of the ``set_gpa_zero_words`` →
        ``halt_and_reset`` cancel path through the GPA SPI bus.

        Runs without a physical gradient board attached. The marga
        serialiser still shifts the registered words onto the SPI
        lines; the busy bit asserts and clears regardless of whether
        anything is wired up on the other end. What we verify from the
        client:

        * The server accepts the zero-words registration.
        * ``halt_and_reset`` returns ``halted=True`` with no errors or
          warnings in the status map. The server's halt_and_reset polls
          the GPA serialiser idle bit after each direct write; any
          ``write_gpa_word_direct`` timeout would surface as a warning
          here (see ``hardware::halt_and_reset`` in marcos_server).
        * Both gradient busy bits (``fhdo_busy``, ``ocra1_busy`` in
          marga register 5 / regstatus.status) are clear once
          ``halt_and_reset`` returns.
        * The registered words persist across multiple
          ``halt_and_reset`` calls without re-registration, matching
          the documented process-lifetime semantics of
          ``_gpa_zero_words``.

        Note: the FHDO/OCRA1 protocol error latch (status_latch & 0x7)
        is *not* asserted here, because with no board on MISO those
        bits may trip spuriously on echo mismatches. End-to-end echo
        correctness needs a real board and is out of scope for this
        runner.
        """
        # Production 4-word vector for GPA-FHDO. The server doesn't
        # interpret the bits during halt_and_reset -- it just shifts
        # them out via write_gpa_word_direct, one per registered
        # channel -- but using the real constant keeps the test aligned
        # with what Experiment.__init__ sends in real use.
        zero_words = list(gb.GPAFHDO._GRAD_ZERO_WORDS)

        result, status = ops.set_gpa_zero_words(zero_words, self.s)
        self.assertEqual(result, 0)
        self.assertEqual(status, {})

        # First halt_and_reset after registration drives the SPI bus.
        halted, hr_status = ops.halt_and_reset(self.s)
        self.assertTrue(halted)
        self.assertEqual(hr_status, {})

        # Both gradient busy bits clear once halt_and_reset returns.
        # Layout: bit 17 = fhdo_busy, bit 16 = ocra1_busy (see marga.sv
        # fld_status assignment).
        regs, _ = ops.regstatus(self.s)
        self.assertEqual(regs.status & 0x30000, 0,
                         "halt_and_reset left FHDO/OCRA1 busy bits asserted")

        # Process-lifetime persistence: a second halt_and_reset without
        # re-registering should still succeed, proving the server still
        # holds the previously-registered vector.
        halted, hr_status = ops.halt_and_reset(self.s)
        self.assertTrue(halted)
        self.assertEqual(hr_status, {})

        # Drain the sticky status latch so subsequent tests (e.g. test_idle)
        # don't see a stale fhdo_err from SPI echo mismatches when no
        # physical gradient board is attached.
        ops.regstatus(self.s)

    @unittest.skip("marga devel")
    def test_acquire_simple(self):
        # For comprehensive tests, see test_loopback.py
        samples = 10
        packet = construct_packet({'acq': samples})
        reply = send_packet(packet, self.s)
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)

        self.assertEqual(reply[:4], (reply_pkt, 1, 0, version_full))
        self.assertEqual(len(acquired_data_raw), samples*8)
        self.assertIs(type(data), np.ndarray)
        self.assertEqual(data.size, samples)

        if False:
            plt.plot(np.abs(data));plt.show()

    @unittest.skip("rewrite needed")
    def test_bad_packet_format(self):
        packet = construct_packet({'configure_hw':
                                   {'lo_freq': 7.12345, # floats instead of uints
                                    'tx_div': 1.234}})
        reply_packet = send_packet(packet, self.s)
        # CONTINUE HERE: this should be handled gracefully by the server
        st()
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full, {'configure_hw': 3}, {}]
        )

    @unittest.skip("comment this line out to shut down the server after testing")
    def test_exit(self): # last in alphabetical order
        reply = ops.close_server(self.s)
        self.assertEqual(reply,
                         (reply_pkt, 1, 0, version_full, {}, {'infos': ['Shutting down server.']}))

def throughput_test(s):
    packet_idx = 0

    for k in range(7):
        msg = msgpack.packb(construct_packet({'test_server_throughput': 10**k}))

        process(send_msg(msg, s))
        packet_idx += 2

def random_test(s):
    # Random other packet
    process(send_msg(msgpack.packb(construct_packet({'boo': 3}) , s)))

def shutdown_server(s):
    msg = msgpack.packb(construct_packet( {}, 0, command=close_server))
    process(send_msg(msg, s), print_all=True)

def test_client(s):
    packet_idx = 0
    pkt = construct_packet( {
        'configure_hw': {
            'fpga_clk_word1': 0x1,
            'fpga_clk_word2': 0x2
            # 'fpga_clk_word3': 0x3,
        },
    }, packet_idx)
    process(send_msg(msgpack.packb(pkt), s), print_all=True)

def main_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        # throughput_test(s)
        test_client(s)
        # shutdown_server(s)

if __name__ == "__main__":
    # main_test()
    unittest.main()
