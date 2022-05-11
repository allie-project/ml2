//! Contains some audio utils from larynx: https://github.com/rhasspy/larynx/blob/master/larynx/audio.py

use std::f64::consts::E;

use ndarray::{s, Array, Array1, Array2, Dimension, NewAxis, Zip};
use ndrustfft::{ndfft_r2c_par, ndifft_r2c_par, R2cFftHandler};
use num_complex::Complex64;
use num_traits::Float;

use crate::ndarray::{diff, maximum, minimum, subtract_outer};

pub mod window;

#[derive(Copy, Clone, Debug)]
pub struct AudioSettings {
	pub filter_length: i16,
	pub hop_length: i16,
	pub win_length: i16,
	pub mel_channels: i8,
	pub sample_rate: i32,
	pub sample_bytes: i8,
	pub channels: i8,
	pub mel_fmin: f64,
	pub mel_fmax: Option<f64>,
	pub ref_level_db: f64,
	pub spec_gain: f64,

	pub signal_norm: bool,
	pub min_level_db: f64,
	pub max_norm: f64,
	pub clip_norm: bool,
	pub symmetric_norm: bool,
	pub do_dynamic_range_compression: bool,
	pub convert_db_to_amp: bool
}

impl Default for AudioSettings {
	fn default() -> Self {
		Self {
			filter_length: 1024,
			hop_length: 256,
			win_length: 256,
			mel_channels: 80,
			sample_rate: 22050,
			sample_bytes: 2,
			channels: 1,
			mel_fmin: 0.0,
			mel_fmax: Some(8000.0),
			ref_level_db: 20.0,
			spec_gain: 1.0,

			signal_norm: false,
			min_level_db: -100.0,
			max_norm: 4.0,
			clip_norm: true,
			symmetric_norm: true,
			do_dynamic_range_compression: true,
			convert_db_to_amp: true
		}
	}
}

impl AudioSettings {
	pub fn amp_to_db(&self, mel_amp: &Array1<f64>) -> Array1<f64> {
		let mut m = mel_amp.clone();
		m.par_map_inplace(|x| *x = x.max(1e-5).log10());
		self.spec_gain * m
	}

	pub fn db_to_amp(&self, mel_db: &Array1<f64>) -> Array1<f64> {
		let mut m = mel_db.clone();
		m.par_map_inplace(|x| *x = 10f64.powf(*x / self.spec_gain));
		m
	}

	pub fn normalize(&self, mel_db: &Array1<f64>) -> Array1<f64> {
		let mel_norm = ((mel_db - self.ref_level_db) - self.min_level_db) / (-self.min_level_db);
		if self.symmetric_norm {
			let mut mel_norm = ((2.0 * self.max_norm) * mel_norm) - self.max_norm;
			if self.clip_norm {
				mel_norm.par_map_inplace(|x| *x = x.max(-self.max_norm).min(self.max_norm));
			}
			mel_norm
		} else {
			let mut mel_norm = self.max_norm * mel_norm;
			if self.clip_norm {
				mel_norm.par_map_inplace(|x| *x = x.max(0.0).min(self.max_norm));
			}
			mel_norm
		}
	}

	pub fn denormalize(&self, mel_db: &Array1<f64>) -> Array1<f64> {
		let mel_denorm = if self.symmetric_norm {
			let mut mel_denorm = mel_db.clone();
			if self.clip_norm {
				mel_denorm.par_map_inplace(|x| *x = x.max(-self.max_norm).min(self.max_norm));
			}

			mel_denorm = ((mel_denorm + self.max_norm) * -self.min_level_db / (2.0 * self.max_norm)) + self.min_level_db;
			mel_denorm
		} else {
			let mut mel_denorm = mel_db.clone();
			if self.clip_norm {
				mel_denorm.par_map_inplace(|x| *x = x.max(0.0).min(self.max_norm));
			}

			mel_denorm = (mel_denorm * -self.min_level_db / self.max_norm) + self.min_level_db;
			mel_denorm
		};

		mel_denorm + self.ref_level_db
	}
}

pub fn float_to_int16(x: &Array1<f64>, max_wav_value: Option<f64>) -> Array1<i16> {
	let max_wav_value = max_wav_value.unwrap_or(32767.0);

	let mut t = x.clone();
	t.par_map_inplace(|x| *x = x.abs());
	let max_abs = t.iter().fold(0.0, |a, b| a.max(*b));
	let audio_norm = x * (max_wav_value / max_abs.max(0.01));
	audio_norm.mapv(|x| x.max(-max_wav_value).min(max_wav_value).round() as i16)
}

pub fn mel_basis(sr: usize, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Array2<f64> {
	let mut weights = Array::zeros((n_mels, (1 + n_fft / 2) as usize));
	let fft_freqs = fft_frequencies(sr, n_fft);
	let mel_f = mel_frequencies(n_mels + 2, fmin, fmax);
	let fdiff = diff(mel_f.view(), None);
	let ramps = subtract_outer(&mel_f, &fft_freqs);
	for i in 0..n_mels {
		let lower = -&ramps.slice(s![i, ..]) / fdiff[i];
		let upper = &ramps.slice(s![i + 2, ..]) / fdiff[i + 1];
		weights
			.slice_mut(s![i, ..])
			.assign(&maximum(Array::zeros(lower.raw_dim()).view(), minimum(lower.view(), upper.view()).view()));
	}
	let enorm = 2.0 / (&mel_f.slice(s![2..n_mels + 2]) - &mel_f.slice(s![..n_mels]));
	weights *= &enorm.slice(s![.., NewAxis]);
	weights
}

pub fn dynamic_range_compression<D>(x: &Array<f64, D>, c: f64, clip_val: f64) -> Array<f64, D>
where
	D: Dimension
{
	let mut x = x.clone();
	x.par_map_inplace(|v| *v = (v.max(clip_val) * c).log(E));
	x
}

pub fn dynamic_range_decompression<D>(x: &Array<f64, D>, c: f64) -> Array<f64, D>
where
	D: Dimension
{
	let mut x = x.clone();
	x.par_map_inplace(|v| *v = v.exp() / c);
	x
}

pub fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64) -> Array1<f64> {
	let min_mel = hz_to_mel(fmin);
	let max_mel = hz_to_mel(fmax);

	let mels = Array::linspace(min_mel, max_mel, n_mels);
	mels_to_hz(&mels)
}

pub fn fft_frequencies(sample_rate: usize, n_fft: usize) -> Array1<f64> {
	let n_fft = n_fft / 2 + 1;
	Array::linspace(0.0, sample_rate as f64 / 2.0, n_fft)
}

pub fn hz_to_mel(frequency: f64) -> f64 {
	// fill in the linear part
	let f_min = 0.0;
	let f_sp = 200.0 / 3.0;

	let mut mel = (frequency - f_min) / f_sp;

	// fill in the log-scale part
	let min_log_hz = 1000.0;
	let min_log_mel = (min_log_hz - f_min) / f_sp;
	let logstep = 6.4f64.log(E) / 27.0;

	if frequency >= min_log_hz {
		mel = min_log_mel + (frequency / min_log_hz).log(E) / logstep
	}

	mel
}

pub fn hz_to_mels(frequencies: &Array1<f64>) -> Array1<f64> {
	// fill in the linear part
	let f_min = 0.0;
	let f_sp = 200.0 / 3.0;

	let mut mels = (frequencies - f_min) / f_sp;

	// fill in the log-scale part
	let min_log_hz = 1000.0;
	let min_log_mel = (min_log_hz - f_min) / f_sp;
	let logstep = 6.4f64.log(E) / 27.0;

	Zip::indexed(mels.view_mut()).par_for_each(|i, v| {
		if frequencies[i] >= min_log_hz {
			*v = min_log_mel + (frequencies[i] / min_log_hz).log(E) / logstep
		}
	});

	mels
}

pub fn mel_to_hz(mel: f64) -> f64 {
	let f_min = 0.0;
	let f_sp = 200.0 / 3.0;
	let mut freq = f_min + f_sp * mel;

	// fill in the log-scale part
	let min_log_hz = 1000.0;
	let min_log_mel = (min_log_hz - f_min) / f_sp;
	let logstep = 6.4f64.log(E) / 27.0;

	if mel >= min_log_mel {
		freq = min_log_hz * (logstep * (mel - min_log_mel)).exp()
	}

	freq
}

pub fn mels_to_hz(mels: &Array1<f64>) -> Array1<f64> {
	let f_min = 0.0;
	let f_sp = 200.0 / 3.0;
	let mut freqs = f_min + f_sp * mels;

	// fill in the log-scale part
	let min_log_hz = 1000.0;
	let min_log_mel = (min_log_hz - f_min) / f_sp;
	let logstep = 6.4f64.log(E) / 27.0;

	Zip::indexed(freqs.view_mut()).par_for_each(|i, v| {
		if mels[i] >= min_log_mel {
			*v = min_log_hz * (logstep * (mels[i] - min_log_mel)).exp()
		}
	});
	freqs
}

/// Compute the one-dimensional discrete Fourier Transform for real input.
pub fn rfft(x: &Array1<f64>) -> Array1<Complex64> {
	let mut v = Array1::<Complex64>::zeros(x.len() / 2 + 1);
	let mut handler = R2cFftHandler::<f64>::new(x.len());
	ndfft_r2c_par(&x.view(), &mut v, &mut handler, 0);
	v
}

/// Computes the inverse of rfft.
///
/// This function computes the inverse of the one-dimensional *n*-point
/// discrete Fourier Transform of real input computed by `rfft`.
///
/// The input is expected to be in the form returned by `rfft`, i.e. the
/// real zero-frequency term followed by the complex positive frequency terms
/// in order of increasing frequency.  Since the discrete Fourier Transform of
/// real input is Hermitian-symmetric, the negative frequency terms are taken
/// to be the complex conjugates of the corresponding positive frequency terms.
pub fn irfft(x: &Array1<Complex64>) -> Array1<f64> {
	let len = (x.len() - 1) * 2;
	let mut v = Array1::<f64>::zeros(len);
	let mut handler = R2cFftHandler::<f64>::new(len);
	ndifft_r2c_par(&x.view(), &mut v, &mut handler, 0);
	v
}

pub fn stft(x: &Array1<f64>, fft_size: usize, hopsamp: usize) -> Array2<Complex64> {
	let window = window::hanning::<f64>(fft_size);
	let mut out = Array2::zeros((((x.len() - fft_size) / hopsamp) + 1, fft_size / 2 + 1));
	for i in (0..x.len() - fft_size).step_by(hopsamp) {
		out.slice_mut(s![i / hopsamp, ..])
			.assign(&rfft(&(&x.slice(s![i..i + fft_size]) * &window)));
	}
	out
}

pub fn istft(x: &Array2<Complex64>, fft_size: usize, hopsamp: usize) -> Array1<f64> {
	let window = window::hanning::<f64>(fft_size);
	let time_slices = x.shape()[0];
	let len_samples = time_slices * hopsamp + fft_size;
	let mut out = Array::zeros(len_samples);
	for (n, i) in (0..out.len() - fft_size).step_by(hopsamp).enumerate() {
		let mut slice = out.slice_mut(s![i..i + fft_size]);
		slice += &(&irfft(&x.slice(s![n, ..]).to_owned()) * &window);
	}
	out
}

#[cfg(test)]
mod tests {
	use approx::assert_relative_eq;
	use ndarray::array;
	use ndarray_npy::read_npy;

	use super::*;

	#[test]
	fn test_mel_basis() {
		let m = mel_basis(22050, 1024, 80, 0.0, 8000.0);
		let larynx_mel: Array2<f64> = read_npy("tests/data/audio/larynx-mel-basis.npy").unwrap();
		assert_relative_eq!(m, larynx_mel);
	}

	#[test]
	fn test_dynamic_range_compression() {
		let sample: Array1<f64> = read_npy("tests/data/audio/sample.npy").unwrap();
		let larynx_compressed: Array1<f64> = read_npy("tests/data/audio/larynx-drc-sample.npy").unwrap();
		let compressed = dynamic_range_compression(&sample, 1.0, 1e-5);
		assert_relative_eq!(compressed, larynx_compressed);

		let decompressed = dynamic_range_decompression(&compressed, 1.0);
		assert_relative_eq!(sample, decompressed, epsilon = 1e-7);
	}

	#[test]
	fn test_mel_frequencies() {
		let freq = mel_frequencies(128, 0.0, 11025.0);
		let larynx_mel: Array1<f64> = read_npy("tests/data/audio/larynx-mel-frequencies.npy").unwrap();
		assert_relative_eq!(freq, larynx_mel);
	}

	#[test]
	fn test_fft_frequencies() {
		let freq = fft_frequencies(22050, 2048);
		let larynx_fft: Array1<f64> = read_npy("tests/data/audio/larynx-fft-frequencies.npy").unwrap();
		assert_relative_eq!(freq, larynx_fft);
	}

	#[test]
	fn test_hz_to_mel() {
		assert_relative_eq!(hz_to_mel(1278.0), 18.567854754553583);
	}

	#[test]
	fn test_mels_to_hz() {
		let mels = array![1., 2., 3., 4., 5.];
		let freqs = mels_to_hz(&mels);
		assert_relative_eq!(freqs, array![66.666667, 133.333334, 200.0, 266.666667, 333.333334], epsilon = 1e-5);
	}

	#[test]
	fn test_rfft() {
		let sample: Array1<f64> = read_npy("tests/data/audio/sample.npy").unwrap();
		let numpy_fft: Array1<Complex64> = read_npy("tests/data/audio/numpy-rfft-sample.npy").unwrap();
		let fft = rfft(&sample);
		assert_relative_eq!(fft, numpy_fft, epsilon = 1e-10);

		// irfft(rfft(x)) == x
		let ifft = irfft(&fft);
		assert_relative_eq!(ifft, sample, epsilon = 1e-14);
	}

	#[test]
	fn test_stft() {
		let sample: Array1<f64> = read_npy("tests/data/audio/sample.npy").unwrap();
		let larynx_st: Array2<Complex64> = read_npy("tests/data/audio/larynx-stft-sample.npy").unwrap();
		let st = stft(&sample, 1024, 256);
		assert_relative_eq!(st, larynx_st, epsilon = 1e-10);
	}

	#[test]
	fn test_istft() {
		let sample: Array1<f64> = read_npy("tests/data/audio/sample.npy").unwrap();
		let larynx_ist: Array1<f64> = read_npy("tests/data/audio/larynx-istft-sample.npy").unwrap();
		let st = stft(&sample, 1024, 256);
		let ist = istft(&st, 1024, 256);
		assert_relative_eq!(ist, /* sample */ larynx_ist, epsilon = 1e-10);
	}
}
