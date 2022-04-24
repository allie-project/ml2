//! Contains some audio utils from larynx: https://github.com/rhasspy/larynx/blob/master/larynx/audio.py

use std::f64::consts::E;

use apodize::hanning_iter;
use ndarray::{parallel::prelude::*, s, Array, Array1, Array2, Dimension, NewAxis, Zip};
use num_complex::Complex64;
use realfft::RealFftPlanner;

use crate::ndarray::{diff, maximum, minimum, subtract_outer};

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

pub fn dynamic_range_decompression<D>(x: &mut Array<f64, D>, c: f64)
where
	D: Dimension
{
	x.par_map_inplace(|x| *x = x.exp() / c);
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
	let mut planner = RealFftPlanner::<f64>::new();
	let fft = planner.plan_fft_forward(x.len());
	// FIXME: consider .collect_into_vec()
	let mut inp = x.par_iter().cloned().collect::<Vec<_>>();
	let mut out = fft.make_output_vec();
	fft.process(&mut inp, &mut out).unwrap();
	Array1::from_vec(out)
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
	let mut planner = RealFftPlanner::<f64>::new();
	let fft = planner.plan_fft_inverse((x.len() - 1) * 2);
	// FIXME: consider .collect_into_vec()
	let mut inp = x.par_iter().cloned().collect::<Vec<_>>();
	let mut out = fft.make_output_vec();
	fft.process(&mut inp, &mut out).unwrap();
	let len = out.len();
	// FIXME: consider .collect_into_vec()
	Array1::from_vec(out.into_par_iter().map(|v| v / len as f64).collect::<Vec<_>>())
}

pub fn stft(x: &Array1<f64>, fft_size: usize, hopsamp: usize) -> Array2<Complex64> {
	let window = Array::from_iter(hanning_iter(fft_size));
	let mut out = Array2::zeros((((x.len() - fft_size) / hopsamp) + 1, fft_size / 2 + 1));
	let mut i = 0;
	while i < x.len() - fft_size {
		out.slice_mut(s![i / hopsamp, ..])
			.assign(&rfft(&(&x.slice(s![i..i + fft_size]) * &window)));
		i += hopsamp;
	}
	out
}

pub fn istft(x: &Array2<Complex64>, fft_size: usize, hopsamp: usize) -> Array1<f64> {
	let window = Array::from_iter(hanning_iter(fft_size));
	let time_slices = x.shape()[0];
	let len_samples = time_slices * hopsamp + fft_size;
	let mut out = Array::zeros(len_samples);
	for (n, i) in (0..x.len() - fft_size).step_by(hopsamp).enumerate() {
		let mut slice = out.slice_mut(s![i..i + fft_size]);
		slice += &(&irfft(&x.slice(s![n, ..]).to_owned()) * &window);
	}
	out
}

#[cfg(test)]
mod tests {
	use approx::assert_relative_eq;
	use ndarray::array;
	use num_complex::Complex;

	use super::*;

	#[test]
	fn test_mel_basis() {
		let m = mel_basis(22050, 1024, 80, 0.0, 8000.0);
		assert_relative_eq!(m.slice(s![0, 0..5]), array![0.0, 0.01552772, 0.02265139, 0.00712367, 0.0], epsilon = 1e-7);
		assert_relative_eq!(m.slice(s![1, 0..5]), array![0.0, 0.0, 0.00420203, 0.01972975, 0.01844937], epsilon = 1e-7);
		assert_relative_eq!(m.slice(s![2, 0..5]), array![0.0, 0.0, 0.0, 0.0, 0.00840405], epsilon = 1e-7);
	}

	#[test]
	fn test_hz_to_mel() {
		assert_relative_eq!(hz_to_mel(1278.0), 18.567854754553583);
	}

	#[test]
	fn test_mels_to_hz() {
		let mels = Array::linspace(0.0, 1000.0, 100);
		let freqs = mels_to_hz(&mels);
		assert_relative_eq!(freqs[0], 0.0);
		assert_relative_eq!(freqs[1], 673.4006734006734);
		assert_relative_eq!(freqs[2], 1429.9623783342638);
	}

	#[test]
	fn test_mel_frequencies() {
		let freq = mel_frequencies(128, 0.0, 11025.0);
		assert_relative_eq!(freq[0], 0.0);
		assert_relative_eq!(freq[4], 104.79914851476966);
		assert_relative_eq!(freq[127], 11025.0);
	}

	#[test]
	fn test_rfft() {
		let x = Array::linspace(0.0, 8.0, 12);
		let fft = rfft(&x);
		assert_relative_eq!(
			fft,
			array![
				Complex::new(48.0, 0.0),
				Complex::new(-4.3636, 16.2853),
				Complex::new(-4.3636, 7.5580),
				Complex::new(-4.3636, 4.3636),
				Complex::new(-4.3636, 2.5193),
				Complex::new(-4.3636, 1.1692),
				Complex::new(-4.3636, 0.0)
			],
			epsilon = 1e-3
		);
	}

	#[test]
	fn test_irfft() {
		let x = Array::linspace(0.0, 8.0, 12);
		let y = Array::from_vec(x.into_par_iter().map(|v| Complex::new(*v, 0.0)).collect::<Vec<_>>());
		let ifft = irfft(&y);
		assert_relative_eq!(
			ifft,
			array![
				4.0,
				-1.6322033083870493,
				-8.074349270001139e-17,
				-0.19156238939459738,
				0.0,
				-0.07708621465041189,
				-8.074349270001139e-17,
				-0.04671117790330436,
				0.0,
				-0.03590798404480271,
				2.0185873175002847e-17,
				-0.03305785123966951,
				2.0185873175002847e-17,
				-0.03590798404480281,
				0.0,
				-0.046711177903304237,
				-8.074349270001139e-17,
				-0.0770862146504119,
				0.0,
				-0.19156238939459738,
				-8.074349270001139e-17,
				-1.632203308387049
			],
			epsilon = 1e-5
		);

		// irfft(rfft(x)) == x
		let ifft = irfft(&rfft(&x));
		assert_relative_eq!(ifft, x, epsilon = 1e-14);
	}

	#[test]
	fn test_stft() {
		let x = Array::linspace(0.0, 8.0, 2048);
		let st = stft(&x, 1024, 512);
		assert_relative_eq!(
			st.slice(s![0, 0..5]),
			array![
				Complex64::new(1022.5002445259, 0.0),
				Complex64::new(-512.7456670332, 242.56513095118),
				Complex64::new(0.6692609891746917, -54.63116427322409),
				Complex64::new(0.2508115673636555, -13.645385610696632),
				Complex64::new(0.13373740173669818, -5.456191766516999)
			],
			epsilon = 1e-6
		);
	}

	#[test]
	fn test_istft() {
		let x = Array::linspace(0.0, 12.0, 2048);
		let st = stft(&x, 1024, 512);
		let ist = istft(&st, 1024, 512);
		assert_relative_eq!(
			ist.slice(s![0..5]),
			array![0.0, 5.213839208544835e-13, 1.6683970642084428e-11, 1.266899190029061e-10, 5.338467801992965e-10],
			epsilon = 1e-12
		);
	}
}
