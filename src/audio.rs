//! Contains some audio utils from larynx: https://github.com/rhasspy/larynx/blob/master/larynx/audio.py

use std::f64::consts::E;

use ndarray::{parallel::prelude::*, Array, Array1, Dimension, Zip};
use num_complex::Complex64;
use realfft::RealFftPlanner;

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
	// let mut planner = FftPlanner::new();
	// let fft = planner.plan_fft(x.len(), FftDirection::Forward);
	// let mut buffer = x.into_par_iter().map(|r| Complex64::new(*r, 0.0)).collect::<Vec<_>>();
	// fft.process(&mut buffer);
	// let v = Array1::from_vec(buffer);
	// v.slice(if v.len() % 2 == 0 { s![..v.len() / 2 + 1] } else { s![..(v.len() + 1) / 2] })
	// 	.to_owned()
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

#[cfg(test)]
mod tests {
	use approx::assert_relative_eq;
	use ndarray::array;
	use num_complex::Complex;

	use super::*;

	#[test]
	fn test_hz_to_mel() {
		assert_relative_eq!(hz_to_mel(1278.0), 18.567854754553583);
	}

	#[test]
	fn test_mel_to_hz() {
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
}
