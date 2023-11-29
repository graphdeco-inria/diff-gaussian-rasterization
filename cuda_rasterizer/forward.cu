/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
/**
 * @brief 计算每个高斯的颜色，转换输入的球形谐波系数为简单的RGB颜色。
 *
 * @param idx 当前处理的高斯索引。
 * @param deg 球形谐波的阶数。
 * @param max_coeffs 最大系数数目。
 * @param means 高斯均值数组。
 * @param campos 相机位置。
 * @param shs 球形谐波系数数组。
 * @param clamped 是否被夹紧标记数组。
 *
 * @return glm::vec3 RGB颜色。
 */
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)




	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
/**
 * @brief 计算每个高斯在2D屏幕空间的协方差矩阵。
 *
 * @param mean 高斯的均值。
 * @param focal_x 相机的X轴焦距。
 * @param focal_y 相机的Y轴焦距。
 * @param tan_fovx 视野X轴正切值。
 * @param tan_fovy 视野Y轴正切值。
 * @param cov3D 3D协方差矩阵。
 * @param viewmatrix 视图矩阵。
 *
 * @return float3 2D协方差矩阵。
 */
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
/**
 * @brief 将每个高斯的缩放和旋转属性转换为世界空间中的3D协方差矩阵。
 *
 * @param scale 缩放值。
 * @param mod 修改器。
 * @param rot 旋转四元数。
 * @param cov3D 3D协方差矩阵。
 */
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
/**
 * @brief 高斯点的预处理函数，为光栅化渲染做准备。
 *
 * @details 这个函数通过一系列步骤处理每个高斯点，为接下来的光栅化渲染做准备：
 *          1. 初始化：
 *             1.1 获取当前线程在CUDA网格中的索引，如果超出处理范围，则直接返回。
 *             1.2 初始化radius和touched tiles数量为0，如果不变，则对应的高斯点不会进一步处理。
 *             1.3 检查高斯点是否在视锥体内，不在视锥体内的点不进行后续处理。
 *          2. 点变换与协方差计算：
 *             2.1 对点进行投影变换，得到投影坐标和2D屏幕空间的协方差矩阵。
 *             2.2 根据缩放和旋转参数计算或使用预先计算的3D协方差矩阵。
 *             2.3 计算2D屏幕空间协方差矩阵，用于后续的屏幕空间范围计算。
 *             2.4 使用EWA算法反转协方差矩阵，以便计算屏幕空间覆盖范围。
 *          3. 计算屏幕空间的像素覆盖范围：
 *             根据2D协方差矩阵计算每个高斯点在屏幕空间的覆盖范围，并确定相应的tile范围。
 *          4. 颜色计算与数据存储：
 *             4.1 如果颜色已预先计算，则使用它们，否则根据球形谐波系数转换为RGB颜色。
 *             4.2 存储辅助数据，如点的深度、半径、屏幕坐标，以及反转的2D协方差和透明度，供后续步骤使用。
 *
 * @note 此预处理步骤对于确定每个高斯点在渲染过程中的关键属性非常重要。
 * 
 * @see computeCov3D：用于计算3D协方差矩阵。
 * @see computeCov2D：用于计算2D屏幕空间协方差矩阵。
 * @see computeColorFromSH：用于将球形谐波系数转换为RGB颜色。
 * @see getRect：用于计算高斯点覆盖的屏幕空间tile的范围。
 * 
 * @return none
 */
__global__ void preprocessCUDA(
    int P, // 高斯点的总数
    int D, // 球形谐波的最大阶数
    int M, // 最大系数数目
    const float* orig_points, // 原始高斯点坐标
    const glm::vec3* scales, // 高斯点的缩放因子
    const float scale_modifier, // 缩放修改因子
    const glm::vec4* rotations, // 高斯点的旋转参数（四元数）
    const float* opacities, // 高斯点的透明度
    const float* shs, // 球形谐波系数
    bool* clamped, // 是否被夹紧标记数组
    const float* cov3D_precomp, // 预计算的3D协方差矩阵
    const float* colors_precomp, // 预计算的颜色值
    const float* viewmatrix, // 视图矩阵
    const float* projmatrix, // 投影矩阵
    const glm::vec3* cam_pos, // 相机位置
    const int W, // 渲染宽度
    const int H, // 渲染高度
    const float tan_fovx, // 视野X轴的正切值
    const float tan_fovy, // 视野Y轴的正切值
    const float focal_x, // 相机的X轴焦距
    const float focal_y, // 相机的Y轴焦距
    int* radii, // 存储计算得到的每个高斯点的半径
    float2* points_xy_image, // 存储计算得到的2D屏幕空间坐标
    float* depths, // 存储计算得到的深度值
    float* cov3Ds, // 存储计算得到的3D协方差矩阵
    float* rgb, // 存储计算得到的颜色值
    float4* conic_opacity, // 存储每个高斯点的锥形不透明度和2D协方差
    const dim3 grid, // CUDA网格尺寸
    uint32_t* tiles_touched, // 影响到的tile数量
    bool prefiltered // 是否进行预过滤
)
{
	// ! 1. 初始化
	// 1.1 获取当前线程在CUDA网格中的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return; // 超出处理范围的线程直接返回


	// 1.2 初始化radius和touched tiles数量为0
	// Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;


    // 1.3 检查高斯点是否在视锥内，如果不在，则不进行后续处理
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;


	// ! 2. 点变换与协方差计算
    // 2.1 对点进行投影变换，获取投影坐标和2D屏幕空间的协方差矩阵
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };


    // 2.2 根据缩放和旋转参数计算或使用预先计算的3D协方差矩阵
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}


	// 2.3 计算2D屏幕空间协方差矩阵，用于后续的屏幕空间范围计算
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);


	// 2.4 使用EWA算法反转协方差矩阵
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };


	// ! 3 计算屏幕空间的像素覆盖范围
	// 计算每个高斯点在屏幕空间的覆盖范围，并确定相应的tile范围
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;


	// ! 4. 颜色计算与数据存储
    // 4.1 计算或使用预先计算的颜色
    // 如果颜色未预先计算，则根据球形谐波系数转换为RGB颜色
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}


    // 4.2 存储辅助数据，以便后续步骤使用，包括点的深度、半径、屏幕坐标和反转的2D协方差及透明度
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; // 点的深度
	radii[idx] = my_radius; // 半径
	points_xy_image[idx] = point_image; // 屏幕坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}




// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
/**
 * @brief 主要光栅化渲染函数，按tile为单位进行处理。
 *
 * @details 这个函数实现了3D Gaussian Splatting算法的核心渲染过程。它的执行过程如下：
 *          1. 初始化和设置：确定当前线程块对应的tile范围和进行相关变量的初始化。
 *             1.1 确定当前线程块的Tile范围：识别当前tile和关联的最小/最大像素范围。
 *             1.2 初始化每个线程的像素位置和状态标志：确定当前线程是否处理有效像素，并设置完成标志。
 *             1.3 加载处理范围：载入要处理的ID范围。
 *             1.4 分配共享内存：为批量提取的高斯点数据分配存储空间。
 *             1.5 初始化渲染辅助变量：设置透明度、贡献者计数等变量。
 * 
 *          2. 批量数据处理：遍历每个批次，直到所有点处理完毕或所有线程完成工作。
 *             2.1 循环遍历每个Batch：检查是否完成渲染，并分批提取高斯点数据。
 * 
 *          3. 像素级渲染计算：对于每个线程块中的每个像素进行光栅化处理。
 *             3.1 遍历当前批次的高斯点：更新贡献者计数。
 *             3.2 计算当前像素的颜色贡献：使用锥形矩阵进行采样，计算每个点对当前像素的影响，并更新颜色和透明度。
 * 
 *          4. 写入渲染结果：将计算出的颜色通道数据和最终的T值存储到输出缓冲区。
 *
 * @note 由于其在3D渲染中的关键作用，这个函数的实现需要特别注意性能优化和内存管理。
 * 
 * @see preprocessCUDA：预处理函数，为renderCUDA的光栅化渲染过程做准备。
 * 
 * @return none
 */
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) renderCUDA (
    const uint2* __restrict__ ranges, // 每个tile的高斯点索引范围
    const uint32_t* __restrict__ point_list, // 高斯点索引列表
    int W, // 渲染宽度
    int H, // 渲染高度
    const float2* __restrict__ points_xy_image, // 高斯点的2D屏幕空间坐标
    const float* __restrict__ features, // 高斯点的特征（颜色等）
    const float4* __restrict__ conic_opacity, // 高斯点的锥形不透明度和2D协方差
    float* __restrict__ final_T, // 最终透明度
    uint32_t* __restrict__ n_contrib, // 贡献到每个像素的高斯点数
    const float* __restrict__ bg_color, // 背景颜色
    float* __restrict__ out_color // 输出颜色
)
{
	// ! 1. 初始化和设置
	// 1.1 确定当前线程块的Tile范围
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	
	// 1.2 初始化每个线程的像素位置和状态标志
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// 检查当前线程是否处理有效像素
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;


	// 1.3 加载处理范围
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;


	// 1.4 分配共享内存
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];


	// 1.5 初始化渲染辅助变量
	// Initialize helper variables
	float T = 1.0f; // 透明度
	uint32_t contributor = 0; // 贡献者计数
	uint32_t last_contributor = 0; // 颜色通道
	float C[CHANNELS] = { 0 };


	// ! 2. 批量数据处理，遍历每个批次，直到所有点处理完毕或者所有线程均完成工作
	// 2.1 循环遍历每个Batch
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 2.1.1 检查是否完成渲染，如果整个块已经完成渲染，则跳出循环
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

        // 2.2 批量提取高斯点数据
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// ! 3. 遍历像素级渲染计算
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// 3.1 更新计数
			// Keep track of current position in range
			contributor++;

			// 3.2 计算当前像素的颜色贡献
			// 3.2.1 使用锥形矩阵进行采样，使用conic matrix进行重采样，计算每个点对当前像素的影响
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// 3.2.2 计算透明度alpha并更新T值
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

            // 3.3 更新颜色和透明度
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			
			T = test_T; // 更新T值，考虑透明度

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}


	// ! 4. 写入渲染结果，将渲染结果写入输出缓冲区
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}