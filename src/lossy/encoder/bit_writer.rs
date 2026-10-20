//! Boolean entropy coder, as libwebp's `VP8BitWriter`.

use super::tables::{NEW_RANGE, NORM};

pub(crate) struct BitWriter {
    range: i32,
    value: i32,
    run: u32,
    nb_bits: i32,
    pub(crate) buf: Vec<u8>,
}

impl BitWriter {
    pub(crate) fn new(expected_size: usize) -> Self {
        Self {
            range: 255 - 1,
            value: 0,
            run: 0,
            nb_bits: -8,
            buf: Vec::with_capacity(expected_size),
        }
    }

    fn flush(&mut self) {
        let s = 8 + self.nb_bits;
        let bits = self.value >> s;
        self.value -= bits << s;
        self.nb_bits -= 8;
        if (bits & 0xff) != 0xff {
            if bits & 0x100 != 0 {
                if let Some(last) = self.buf.last_mut() {
                    *last = last.wrapping_add(1);
                }
            }
            if self.run > 0 {
                let value = if bits & 0x100 != 0 { 0x00 } else { 0xff };
                for _ in 0..self.run {
                    self.buf.push(value);
                }
                self.run = 0;
            }
            self.buf.push(bits as u8);
        } else {
            self.run += 1;
        }
    }

    #[inline]
    pub(crate) fn put_bit(&mut self, bit: bool, prob: u8) -> bool {
        let split = (self.range * i32::from(prob)) >> 8;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            let shift = NORM[(self.range & 127) as usize];
            self.range = i32::from(NEW_RANGE[(self.range & 127) as usize]);
            self.value <<= shift;
            self.nb_bits += i32::from(shift);
            if self.nb_bits > 0 {
                self.flush();
            }
        }
        bit
    }

    #[inline]
    pub(crate) fn put_bit_uniform(&mut self, bit: bool) -> bool {
        let split = self.range >> 1;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            self.range = i32::from(NEW_RANGE[(self.range & 127) as usize]);
            self.value <<= 1;
            self.nb_bits += 1;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
        bit
    }

    pub(crate) fn put_bits(&mut self, value: u32, nb_bits: u32) {
        let mut mask = 1u32 << (nb_bits - 1);
        while mask != 0 {
            self.put_bit_uniform(value & mask != 0);
            mask >>= 1;
        }
    }

    pub(crate) fn put_signed_bits(&mut self, value: i32, nb_bits: u32) {
        if !self.put_bit_uniform(value != 0) {
            return;
        }
        if value < 0 {
            self.put_bits((((-value) << 1) | 1) as u32, nb_bits + 1);
        } else {
            self.put_bits((value << 1) as u32, nb_bits + 1);
        }
    }

    pub(crate) fn finish(&mut self) {
        self.put_bits(0, (9 - self.nb_bits) as u32);
        self.nb_bits = 0;
        self.flush();
    }
}
