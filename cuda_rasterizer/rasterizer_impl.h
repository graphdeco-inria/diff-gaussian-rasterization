#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

namespace CudaRasterizer
{
	class RasterizerImpl : public Rasterizer
	{
	private:
		int maxP = 0;
		int maxPixels = 0;
		int resizeMultiplier = 2;

		// Initial aux structs
		size_t sorting_size;
		size_t list_sorting_size;
		size_t scan_size;
		thrust::device_vector<float> depths;
		thrust::device_vector<uint32_t> tiles_touched;
		thrust::device_vector<uint32_t> point_offsets;
		thrust::device_vector<uint64_t> point_list_keys_unsorted;
		thrust::device_vector<uint64_t> point_list_keys;
		thrust::device_vector<uint32_t> point_list_unsorted;
		thrust::device_vector<uint32_t> point_list;
		thrust::device_vector<char> scanning_space;
		thrust::device_vector<char> list_sorting_space;
		thrust::device_vector<bool> clamped;
		thrust::device_vector<int> internal_radii;

		// Internal state kept across forward / backward
		thrust::device_vector<uint2> ranges;
		thrust::device_vector<uint32_t> n_contrib;
		thrust::device_vector<float> accum_alpha;

		thrust::device_vector<float2> means2D;
		thrust::device_vector<float> cov3D;
		thrust::device_vector<float4> conic_opacity;
		thrust::device_vector<float> rgb;

	public:

		virtual void markVisible(
			int P, 
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present) override;

		virtual void forward(
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii) override;

		virtual void backward(
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot) override;

		RasterizerImpl(int resizeMultiplier);

		virtual ~RasterizerImpl() override;
	};
};