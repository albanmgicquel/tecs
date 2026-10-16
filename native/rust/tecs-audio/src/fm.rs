//! An emulated OPL2 or OPL3 that plays as one voice of the mixer.
//!
//! The chip is the YM3812 or the YMF262 of [`ymfm_oxide`], a pure-Rust port of
//! ymfm, and what it is driven by is the scheduler the General MIDI
//! synthesizer uses: the same bounded ring, the same schedule sorted by frame
//! and then by sending, the same source the mixer reads. What differs is the
//! grid. A register write takes effect on the very sample it is due at, so the
//! target's block is one frame and an event lands on the first chip sample at
//! or after its time, never later. Writes due on one sample all apply before
//! it is generated, so a key-off and a key-on there are no change at all: a
//! note is struck again by spacing them a sample apart.
//!
//! The chip runs at its own rate, [`RATE`], and the voice that reads it
//! resamples to the output, as it would a clip recorded at that rate. The
//! clock a game schedules against counts chip samples.
//!
//! ```
//! use tecsaudio::fm::{self, Kind};
//! use tecsaudio::source::Source;
//!
//! let (mut port, mut source) = fm::open(Kind::Opl2);
//! // Key channel 0 on a quarter of a second in, on the chip's own clock.
//! assert!(port.write(0xb0, 0x32, 0.25));
//! let mut out = [0.0f32; 2];
//! source.read(&mut out);
//! ```

use ymfm_oxide::opl::{Ym3812, Ymf262};
use ymfm_oxide::{NullInterface, Output};

use crate::synth::{self, Event, Port, SynthSource, Target, KIND_MESSAGE, KIND_RESET};

/// The input clock of an OPL2, the NTSC colour burst a sound card divides
/// down: 3.579545 MHz. The OPL3 takes four times it.
pub const OPL2_CLOCK: u32 = 3_579_545;

/// The input clock of an OPL3: 14.31818 MHz.
pub const OPL3_CLOCK: u32 = 4 * OPL2_CLOCK;

/// The rate the chip renders at and its clock counts in, in hertz: the input
/// clock over 72 on the OPL2 and over 288 on the OPL3, 49 715.9 Hz, which the
/// emulator truncates to a whole rate.
pub const RATE: u32 = 49_715;

/// Selects which chip to emulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The YM3812: nine channels of two operators, four waveforms, mono.
    Opl2,
    /// The YMF262: eighteen channels that pair into 4-operator voices, eight
    /// waveforms, and stereo from the pan bits of 0xC0 to 0xC8.
    Opl3,
}

impl Kind {
    /// Returns the highest register address the chip decodes: 0xFF on the
    /// OPL2, and 0x1FF on the OPL3, whose second bank is 0x100 up.
    pub const fn last_register(self) -> u16 {
        match self {
            Kind::Opl2 => 0xff,
            Kind::Opl3 => 0x1ff,
        }
    }
}

/// One emulated chip, written by register and read as a stereo pair.
#[derive(Clone)]
enum Chip {
    Opl2(Ym3812),
    Opl3(Ymf262),
}

impl Chip {
    /// Creates a chip in its power-on state.
    fn new(kind: Kind) -> Chip {
        match kind {
            Kind::Opl2 => Chip::Opl2(Ym3812::new(NullInterface)),
            Kind::Opl3 => Chip::Opl3(Ymf262::new(NullInterface)),
        }
    }

    fn kind(&self) -> Kind {
        match self {
            Chip::Opl2(_) => Kind::Opl2,
            Chip::Opl3(_) => Kind::Opl3,
        }
    }

    /// Reports the rate the emulator gives the chip's input clock, which
    /// [`RATE`] has to be.
    #[cfg(test)]
    fn sample_rate(&self) -> u32 {
        match self {
            Chip::Opl2(chip) => chip.sample_rate(OPL2_CLOCK),
            Chip::Opl3(chip) => chip.sample_rate(OPL3_CLOCK),
        }
    }

    /// Writes `value` to `register` the way a driver does, through the address
    /// port and then the data port.
    ///
    /// An OPL3 register from 0x100 up goes through the second bank's address
    /// port (offset 2, data at 3), and the chip itself masks it back to the
    /// first bank until the NEW bit in 0x105 is set, as the hardware does.
    fn write(&mut self, register: u16, value: u8) {
        match self {
            Chip::Opl2(chip) => {
                chip.write(0, register as u8);
                chip.write(1, value);
            }
            Chip::Opl3(chip) => {
                let bank = u32::from(register >> 8 & 1) * 2;
                chip.write(bank, register as u8);
                chip.write(bank + 1, value);
            }
        }
    }

    /// Generates the next sample as a left and right pair: the OPL2's one
    /// output on both sides, the OPL3's outputs A and B, which the pan bits of
    /// each channel select.
    fn generate(&mut self) -> (i32, i32) {
        match self {
            Chip::Opl2(chip) => {
                let mut out = [Output::<1>::default()];
                chip.generate(&mut out);
                (out[0].data[0], out[0].data[0])
            }
            Chip::Opl3(chip) => {
                let mut out = [Output::<4>::default()];
                chip.generate(&mut out);
                (out[0].data[0], out[0].data[1])
            }
        }
    }

    /// Returns the chip to its power-on state, as `new` makes it.
    fn reset(&mut self) {
        *self = Chip::new(self.kind());
    }
}

/// Writes the queue holds between the game thread and the callback, which is
/// also the largest batch `writes` can ever accept.
pub const QUEUE_WRITES: usize = synth::QUEUE_EVENTS;

/// One register write, as a batch crosses the C ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterWrite {
    /// The register, from 0x00 to 0xFF on the OPL2 and 0x000 to 0x1FF on the
    /// OPL3, whose second bank is 0x100 up.
    pub register: u16,
    /// The value, from 0 to 255. Sixteen bits wide so the struct has no
    /// padding for two languages to disagree about.
    pub value: u16,
}

/// The chip as a scheduler target.
///
/// A write event carries the register's low byte in `status`, its bank in
/// `data1` and the value in `data2`.
pub struct Fm {
    chip: Chip,
}

impl Target for Fm {
    const BLOCK_FRAMES: usize = 1;

    fn apply(&mut self, event: &Event) {
        match event.kind {
            KIND_MESSAGE => {
                let register = u16::from(event.status) | (u16::from(event.data1) << 8);
                self.chip.write(register, event.data2);
            }
            KIND_RESET => self.chip.reset(),
            _ => {}
        }
    }

    fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (a, b) = self.chip.generate();
            *l = a as f32 / 32768.0;
            *r = b as f32 / 32768.0;
        }
    }
}

/// The game side of one chip: where register writes are sent and the clock
/// read.
pub struct FmPort {
    port: Port,
    kind: Kind,
}

impl FmPort {
    /// Converts a time on the chip clock to the first sample at or after it.
    ///
    /// A time within a millionth of a sample of a whole sample is that
    /// sample, so a time computed as `now() + k / RATE` lands on sample `k`
    /// rather than one past it for a rounding error. Zero, a negative time and
    /// a time that is not a number all mean the next sample.
    fn frame_at(&self, seconds: f64) -> u64 {
        if seconds.is_nan() || seconds <= 0.0 {
            return 0;
        }
        let exact = seconds * f64::from(self.port.sample_rate());
        let nearest = exact.round();
        if (exact - nearest).abs() < 1e-6 {
            nearest as u64
        } else {
            exact.ceil() as u64
        }
    }

    fn event(register: u16, value: u8, frame: u64) -> Event {
        Event {
            frame,
            kind: KIND_MESSAGE,
            status: register as u8,
            data1: (register >> 8) as u8,
            data2: value,
        }
    }

    /// Reports which chip the port drives.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Queues one register write for `at` seconds on the chip clock, and
    /// reports whether it was queued: false when the queue is full or the
    /// register is past the chip's last.
    pub fn write(&mut self, register: u16, value: u8, at: f64) -> bool {
        if register > self.kind.last_register() {
            return false;
        }
        let frame = self.frame_at(at);
        self.port.push(FmPort::event(register, value, frame))
    }

    /// Queues every write in `writes` for `at`, in order, all or none, and
    /// reports which: none when the queue cannot take them all, or when any
    /// register is past the chip's last or any value past 255.
    pub fn writes(&mut self, writes: &[RegisterWrite], at: f64) -> bool {
        let last = self.kind.last_register();
        if writes
            .iter()
            .any(|write| write.register > last || write.value > 0xff)
        {
            return false;
        }
        let frame = self.frame_at(at);
        self.port.push_all(writes.len(), |index| {
            FmPort::event(writes[index].register, writes[index].value as u8, frame)
        })
    }

    /// Queues a return to the chip's power-on state for `at` seconds on the
    /// chip clock, where zero or a time already passed means the next sample.
    ///
    /// Writes sent before the reset and due before `at` still apply; those
    /// due at or after it are discarded, so nothing sent earlier plays over
    /// the silence the reset makes. Writes sent after it apply as usual.
    pub fn reset(&mut self, at: f64) -> bool {
        let frame = self.frame_at(at);
        self.port.push(Event {
            frame,
            kind: KIND_RESET,
            status: 0,
            data1: 0,
            data2: 0,
        })
    }

    /// Reports the chip clock in seconds: the samples it has rendered over
    /// [`RATE`].
    pub fn now(&self) -> f64 {
        self.port.now()
    }

    /// Reports the samples the chip has rendered.
    pub fn frames(&self) -> u64 {
        self.port.frames()
    }

    /// Reports how many sends the queue refused: a refused batch counts once.
    pub fn dropped(&self) -> u64 {
        self.port.dropped()
    }
}

/// Builds the two halves of one chip: the port the game writes through and
/// the source the mixer reads.
pub fn open(kind: Kind) -> (FmPort, SynthSource<Fm>) {
    let (port, source) = synth::assemble(
        Fm {
            chip: Chip::new(kind),
        },
        RATE,
    );
    (FmPort { port, kind }, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Source;

    /// The writes that put a held sine on channel 0 and key it on. Its
    /// envelope jumps straight to full and holds, so the note sounds the same
    /// wherever the chip's envelope counter happens to stand.
    const NOTE: [(u16, u8); 8] = [
        (0x20, 0x01),
        (0x23, 0x21),
        (0x40, 0x3f),
        (0x43, 0x00),
        (0x63, 0xf0),
        (0x83, 0x0f),
        (0xa0, 0x41),
        (0xb0, 0x32),
    ];

    /// Reads `frames` from a source in pieces of `piece`, as a voice does,
    /// and returns the left channel.
    fn pull(source: &mut impl Source, frames: usize, piece: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames);
        let mut staging = vec![0.0f32; piece * 2];
        while out.len() < frames {
            let wanted = (frames - out.len()).min(piece);
            let got = source.read(&mut staging[..wanted * 2]);
            out.extend(
                staging[..got * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|frame| frame[0]),
            );
        }
        out
    }

    /// Renders `frames` straight from a chip, applying each write before the
    /// sample it names, which is what ymfm's renders mean by a write's time.
    fn direct(kind: Kind, writes: &[(u64, u16, u8)], frames: usize) -> Vec<f32> {
        let mut chip = Chip::new(kind);
        let mut next = 0;
        (0..frames as u64)
            .map(|frame| {
                while next < writes.len() && writes[next].0 <= frame {
                    chip.write(writes[next].1, writes[next].2);
                    next += 1;
                }
                chip.generate().0 as f32 / 32768.0
            })
            .collect()
    }

    fn timed(frame: u64) -> Vec<(u64, u16, u8)> {
        NOTE.iter()
            .map(|(register, value)| (frame, *register, *value))
            .collect()
    }

    #[test]
    fn lands_a_write_on_the_first_sample_at_or_after_its_time() {
        let rate = f64::from(RATE);
        for (at, frame) in [
            (1000.0 / rate, 1000),
            (1000.4 / rate, 1001),
            (1000.9 / rate, 1001),
        ] {
            let (mut port, mut source) = open(Kind::Opl2);
            for (register, value) in NOTE {
                assert!(port.write(register, value, at));
            }
            // A piece size that does not divide anything, so the reads fall
            // across the write's sample at an arbitrary offset.
            let out = pull(&mut source, 3000, 37);
            assert_eq!(out, direct(Kind::Opl2, &timed(frame), 3000), "at {at}");
            assert!(out[..frame as usize].iter().all(|sample| *sample == 0.0));
            assert!(out[frame as usize + 4..]
                .iter()
                .any(|sample| *sample != 0.0));
        }
    }

    #[test]
    fn lands_a_time_computed_from_the_clock_on_its_sample() {
        let (mut port, mut source) = open(Kind::Opl2);
        pull(&mut source, 777, 64);
        // `now` and a step of whole samples, each as floating point carries it.
        let at = port.now() + 123.0 / f64::from(RATE);
        for (register, value) in NOTE {
            assert!(port.write(register, value, at));
        }
        let out = pull(&mut source, 2000, 64);
        let reference = direct(Kind::Opl2, &timed(123), 2000);
        assert_eq!(out, reference);
    }

    #[test]
    fn applies_writes_due_on_one_sample_in_sending_order() {
        let (mut port, mut source) = open(Kind::Opl2);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        // Two frequencies for the same sample: the second one sent wins.
        let at = 500.0 / f64::from(RATE);
        assert!(port.write(0xa0, 0x10, at));
        assert!(port.write(0xa0, 0x80, at));
        let out = pull(&mut source, 1500, 256);
        let mut writes = timed(0);
        writes.push((500, 0xa0, 0x10));
        writes.push((500, 0xa0, 0x80));
        assert_eq!(out, direct(Kind::Opl2, &writes, 1500));
        writes.truncate(NOTE.len() + 1);
        assert_ne!(out, direct(Kind::Opl2, &writes, 1500));
    }

    #[test]
    fn orders_writes_by_time_rather_than_by_sending() {
        let (mut port, mut source) = open(Kind::Opl2);
        let rate = f64::from(RATE);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        assert!(port.write(0xb0, 0x12, 900.0 / rate));
        assert!(port.write(0xa0, 0x99, 300.0 / rate));
        let out = pull(&mut source, 1500, 100);
        let mut writes = timed(0);
        writes.push((300, 0xa0, 0x99));
        writes.push((900, 0xb0, 0x12));
        assert_eq!(out, direct(Kind::Opl2, &writes, 1500));
    }

    #[test]
    fn applies_a_past_or_untimed_write_on_the_next_sample() {
        let (mut port, mut source) = open(Kind::Opl2);
        pull(&mut source, 640, 64);
        assert!(port.write(NOTE[0].0, NOTE[0].1, 0.0));
        assert!(port.write(NOTE[1].0, NOTE[1].1, -3.0));
        for (register, value) in &NOTE[2..] {
            assert!(port.write(*register, *value, 1.0 / f64::from(RATE)));
        }
        let out = pull(&mut source, 1000, 64);
        let reference = direct(Kind::Opl2, &timed(0), 1000);
        assert_eq!(out, reference);
    }

    #[test]
    fn plays_the_opl2_on_both_sides_and_the_opl3_from_its_pan_bits() {
        let (mut port, mut source) = open(Kind::Opl2);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        let mut out = vec![0.0f32; 2 * 600];
        let mut done = 0;
        while done < 600 {
            done += source.read(&mut out[done * 2..]);
        }
        assert!(out
            .as_chunks::<2>()
            .0
            .iter()
            .all(|frame| frame[0] == frame[1]));
        assert!(out.iter().any(|sample| *sample != 0.0));

        let (mut port, mut source) = open(Kind::Opl3);
        assert!(port.write(0x105, 0x01, 0.0));
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        // Channel 0 to the left only.
        assert!(port.write(0xc0, 0x10, 0.0));
        let mut out = vec![0.0f32; 2 * 600];
        let mut done = 0;
        while done < 600 {
            done += source.read(&mut out[done * 2..]);
        }
        assert!(out.as_chunks::<2>().0.iter().any(|frame| frame[0] != 0.0));
        assert!(out.as_chunks::<2>().0.iter().all(|frame| frame[1] == 0.0));
    }

    #[test]
    fn refuses_a_register_past_the_chip_and_queues_nothing() {
        let (mut opl2, _source) = open(Kind::Opl2);
        assert!(!opl2.write(0x100, 0x01, 0.0));
        assert!(opl2.write(0xff, 0x01, 0.0));
        let (mut opl3, _source) = open(Kind::Opl3);
        assert!(opl3.write(0x1ff, 0x01, 0.0));
        assert!(!opl3.write(0x200, 0x01, 0.0));
        let bad = [
            RegisterWrite {
                register: 0x20,
                value: 1,
            },
            RegisterWrite {
                register: 0x40,
                value: 0x100,
            },
        ];
        assert!(!opl3.writes(&bad, 0.0));
        assert_eq!(
            opl2.dropped() + opl3.dropped(),
            0,
            "a refused value is not a drop"
        );
    }

    #[test]
    fn bounds_the_queue_and_takes_a_batch_whole_or_not_at_all() {
        let (mut port, mut source) = open(Kind::Opl2);
        let later = 1.0;
        let batch: Vec<RegisterWrite> = (0..100)
            .map(|index| RegisterWrite {
                register: 0xa0,
                value: index,
            })
            .collect();
        for _ in 0..QUEUE_WRITES - 50 {
            assert!(port.write(0x20, 0x01, later));
        }
        // Fifty slots are left: the batch of a hundred does not fit and none
        // of it is queued, so all fifty singles still are.
        assert!(!port.writes(&batch, later));
        for _ in 0..50 {
            assert!(port.write(0x20, 0x01, later));
        }
        assert!(!port.write(0x20, 0x01, later));
        assert_eq!(port.dropped(), 2);

        // One sample moves the ring into the schedule, which has the room.
        pull(&mut source, 1, 1);
        assert!(port.writes(&batch, later));
        let too_many = vec![
            RegisterWrite {
                register: 0x20,
                value: 1,
            };
            QUEUE_WRITES + 1
        ];
        let (mut fresh, _source) = open(Kind::Opl2);
        assert!(
            !fresh.writes(&too_many, 0.0),
            "a batch past the queue never fits"
        );
        assert!(fresh.writes(&too_many[..QUEUE_WRITES], 0.0));
    }

    #[test]
    fn resets_to_power_on_and_discards_what_was_pending() {
        let (mut port, mut source) = open(Kind::Opl3);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        pull(&mut source, 1000, 128);
        // A note scheduled for later, then a reset sent after it.
        assert!(port.write(0xb1, 0x32, 2000.0 / f64::from(RATE)));
        assert!(port.reset(0.0));
        let after = pull(&mut source, 3000, 128);
        assert!(
            after.iter().all(|sample| *sample == 0.0),
            "the chip is silent and the note gone"
        );

        // What follows a reset plays on a chip just as a new one would.
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        let replay = pull(&mut source, 2000, 128);
        assert_eq!(replay, direct(Kind::Opl3, &timed(0), 2000));
    }

    #[test]
    fn resets_at_its_time_keeping_what_falls_before_it() {
        let rate = f64::from(RATE);
        let (mut port, mut source) = open(Kind::Opl2);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        // A frequency change before the reset, one after it, then the reset
        // between them, and a write sent after the reset for later still.
        assert!(port.write(0xa0, 0x80, 400.0 / rate));
        assert!(port.write(0xa0, 0x20, 900.0 / rate));
        assert!(port.reset(600.0 / rate));
        for (register, value) in NOTE {
            assert!(port.write(register, value, 800.0 / rate));
        }
        let out = pull(&mut source, 1200, 50);
        let mut writes = timed(0);
        writes.push((400, 0xa0, 0x80));
        let before = direct(Kind::Opl2, &writes, 600);
        assert_eq!(&out[..600], &before[..]);
        assert!(
            out[600..800].iter().all(|sample| *sample == 0.0),
            "the reset silences the chip at its sample"
        );
        // The note sent after the reset plays, and the frequency change at 900
        // sent before it does not.
        let after = direct(Kind::Opl2, &timed(0), 400);
        assert_eq!(&out[800..], &after[..]);
    }

    #[test]
    fn renders_at_the_rate_the_emulator_gives_each_chip_clock() {
        assert_eq!(Chip::new(Kind::Opl2).sample_rate(), RATE);
        assert_eq!(Chip::new(Kind::Opl3).sample_rate(), RATE);
    }

    #[test]
    fn writes_the_second_bank_of_an_opl3_through_its_own_port() {
        let (mut port, mut source) = open(Kind::Opl3);
        assert!(port.write(0x105, 0x01, 0.0));
        // The note on channel 9, the first of the second bank, centred.
        for (register, value) in NOTE {
            assert!(port.write(0x100 | register, value, 0.0));
        }
        assert!(port.write(0x1c0, 0x31, 0.0));
        let out = pull(&mut source, 600, 64);
        assert!(out.iter().any(|sample| *sample != 0.0));
        // Without the NEW bit the chip masks the bank away, so the same
        // writes land on channel 0 of the first bank.
        let (mut port, mut source) = open(Kind::Opl3);
        for (register, value) in NOTE {
            assert!(port.write(0x100 | register, value, 0.0));
        }
        let out = pull(&mut source, 600, 64);
        assert!(out.iter().any(|sample| *sample != 0.0));
        assert_eq!(out, direct(Kind::Opl3, &timed(0), 600));
    }

    #[test]
    fn takes_a_key_off_and_on_due_on_one_sample_as_no_change() {
        let rate = f64::from(RATE);
        let (mut port, mut source) = open(Kind::Opl2);
        for (register, value) in NOTE {
            assert!(port.write(register, value, 0.0));
        }
        // Released and keyed again on sample 500: both apply before it, so
        // the note holds as if neither had been sent.
        assert!(port.write(0xb0, 0x12, 500.0 / rate));
        assert!(port.write(0xb0, 0x32, 500.0 / rate));
        let out = pull(&mut source, 1500, 64);
        assert_eq!(out, direct(Kind::Opl2, &timed(0), 1500));
    }

    #[test]
    fn runs_the_clock_in_chip_samples() {
        let (port, mut source) = open(Kind::Opl2);
        assert_eq!(port.now(), 0.0);
        pull(&mut source, RATE as usize / 2, 512);
        assert_eq!(port.frames(), u64::from(RATE / 2));
        assert!((port.now() - f64::from(RATE / 2) / f64::from(RATE)).abs() < 1e-12);
        assert_eq!(source.sample_rate(), RATE);
        assert_eq!(source.channels(), 2);
        assert!(!source.exhausted());
    }
}
