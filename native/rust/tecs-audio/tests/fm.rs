//! An emulated OPL as a voice of the mixer, with no audio device.
//!
//! The emulator is the ymfm-oxide crate, which holds itself to ymfm sample for
//! sample, and the scheduling grid is tested in `src/fm.rs`; this covers what lies between them and the output: the parked
//! source a `PLAY` takes, resampling from the chip's rate, the voice gain, and
//! a clock that stops with the voice.

use tecsaudio::command::{self, Command};
use tecsaudio::engine::Engine;
use tecsaudio::fm::{FmPort, Kind, RegisterWrite, RATE as CHIP_RATE};

const RATE: u32 = 48_000;

/// A held sine on channel 0 at 440 Hz, keyed on.
const NOTE: [(u16, u16); 8] = [
    (0x20, 0x01),
    (0x23, 0x21),
    (0x40, 0x3f),
    (0x43, 0x00),
    (0x63, 0xf0),
    (0x83, 0x0f),
    (0xa0, 0x41),
    (0xb0, 0x32),
];

fn command(kind: u32, handle: u32) -> Command {
    Command {
        kind,
        handle,
        clip: 0,
        flags: 0,
        gain: 1.0,
        pitch: 1.0,
        start: 0.0,
        fade: 0.0,
        loop_start: 0.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn started(kind: Kind, gain: f32) -> (Engine, FmPort) {
    let engine = Engine::open_offline(RATE, 2, 8);
    let port = engine.open_fm(1, kind).expect("the chip opens");
    let mut play = command(command::PLAY, 1);
    play.flags = command::FLAG_SYNTH;
    play.gain = gain;
    engine.submit(&[play]);
    assert_eq!(engine.voice_count(), 1);
    (engine, port)
}

fn note() -> Vec<RegisterWrite> {
    NOTE.iter()
        .map(|(register, value)| RegisterWrite {
            register: *register,
            value: *value,
        })
        .collect()
}

fn render(engine: &Engine, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * 2];
    for chunk in out.chunks_mut(512 * 2) {
        engine.render(chunk);
    }
    out
}

fn onset(out: &[f32]) -> Option<usize> {
    out.iter()
        .position(|sample| *sample != 0.0)
        .map(|index| index / 2)
}

#[test]
fn starts_nothing_for_an_fm_play_with_no_chip_parked() {
    let engine = Engine::open_offline(RATE, 2, 8);
    let mut play = command(command::PLAY, 9);
    play.flags = command::FLAG_SYNTH;
    engine.submit(&[play]);
    assert_eq!(engine.voice_count(), 0);
}

#[test]
fn plays_a_scheduled_note_resampled_to_the_output() {
    let (engine, mut port) = started(Kind::Opl2, 1.0);
    assert_eq!(port.now(), 0.0);
    // A tenth of a second in, on the chip clock.
    assert!(port.writes(&note(), 0.1));
    let out = render(&engine, 9600);
    let start = onset(&out).expect("the note sounds");
    // The chip lands the note on sample 4972 of its own clock, which the
    // voice reaches at output frame 4972 * 48000 / 49715, give or take the
    // resampler's one frame of interpolation.
    let expected = (4972.0 * f64::from(RATE) / f64::from(CHIP_RATE)) as usize;
    assert!(
        (expected..=expected + 2).contains(&start),
        "the note starts at {start}"
    );
    // Mono on both sides.
    assert!(out
        .as_chunks::<2>()
        .0
        .iter()
        .all(|frame| frame[0] == frame[1]));
    // The clock is chip samples, running at most a frame or two ahead of the
    // mix it fed.
    let rendered = port.frames() as f64 / f64::from(CHIP_RATE);
    assert!(
        (rendered - 9600.0 / f64::from(RATE)).abs() < 3.0 / f64::from(RATE),
        "{rendered}"
    );
}

#[test]
fn renders_the_same_schedule_to_the_same_samples() {
    let mut takes = Vec::new();
    for _ in 0..2 {
        let (engine, mut port) = started(Kind::Opl3, 1.0);
        // NEW on, and channel 0 panned to both sides, which with NEW on it
        // has to be to sound at all.
        assert!(port.write(0x105, 0x01, 0.0));
        assert!(port.write(0xc0, 0x30, 0.0));
        for step in 0..4 {
            let at = f64::from(step) * 0.05;
            assert!(port.writes(&note(), at));
            assert!(port.write(0xa0, 0x41 + step as u8 * 0x20, at + 0.01));
            assert!(port.write(0xb0, 0x12, at + 0.04));
        }
        takes.push(render(&engine, RATE as usize / 4));
    }
    assert!(takes[0].iter().any(|sample| *sample != 0.0));
    assert_eq!(takes[0], takes[1]);
}

#[test]
fn scales_the_chip_by_its_voice_gain() {
    let mut levels = Vec::new();
    for gain in [1.0f32, 0.5, 0.0] {
        let (engine, mut port) = started(Kind::Opl2, 1.0);
        let mut set = command(command::SET_GAIN, 1);
        set.gain = gain;
        engine.submit(&[set]);
        assert!(port.writes(&note(), 0.0));
        levels.push(render(&engine, 4800));
    }
    assert!(levels[0].iter().any(|sample| *sample != 0.0));
    for (full, half) in levels[0].iter().zip(levels[1].iter()) {
        assert!((full * 0.5 - half).abs() < 1e-6);
    }
    assert!(levels[2].iter().all(|sample| *sample == 0.0));
}

#[test]
fn holds_the_clock_while_the_voice_is_paused() {
    let (engine, port) = started(Kind::Opl2, 1.0);
    render(&engine, 1024);
    let before = port.frames();
    assert!(before > 0);
    engine.submit(&[command(command::PAUSE, 1)]);
    render(&engine, 4800);
    assert_eq!(port.frames(), before, "a paused voice does not advance");
    engine.submit(&[command(command::RESUME, 1)]);
    render(&engine, 1024);
    assert!(port.frames() > before);
}
