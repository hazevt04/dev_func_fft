#pragma once

class FFT_Params {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int warp = 32;
};


class FFT_64_forward : public FFT_Params {
	public:
	static const int fft_exp = 6;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

