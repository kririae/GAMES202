#include "denoiser.h"

#include <omp.h>

#include <cassert>

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
  int height = m_accColor.m_height;
  int width = m_accColor.m_width;
  Matrix4x4 preWorldToScreen =
      m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
  Matrix4x4 preWorldToCamera =
      m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];

#pragma omp parallel for default(none) shared(height, width, frameInfo, preWorldToScreen)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      m_valid(x, y) = false;
      m_misc(x, y) = Float3(0.0f);

      const int obj_id = static_cast<int>(frameInfo.m_id(x, y));
      if (obj_id < 0) continue;

      const auto &M_p_1 = Inverse(frameInfo.m_matrix[obj_id]);
      auto P = frameInfo.m_position(x, y);
      P = M_p_1(P, Float3::EType::Point);

      const auto &Ps = preWorldToScreen(P, Float3::EType::Point);
      // const auto &Pc = preWorldToCamera(P, Float3::EType::Point);
      if (0 <= Ps.x && Ps.x < width && 0 <= Ps.y && Ps.y < height &&
          int(m_preFrameInfo.m_id(int(Ps.x), int(Ps.y))) == obj_id) {
        m_valid(x, y) = true;
        m_misc(x, y) = m_accColor(Ps.x, Ps.y);
      }
    }
  }

  std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
  int height = m_accColor.m_height;
  int width = m_accColor.m_width;
  constexpr int kernelRadius = 3;

#pragma omp parallel for default(none) shared(height, width, curFilteredColor)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int cnt = 0;
      Float3 E, E2, Var;
      Float3 color = m_accColor(x, y);

      for (int i = -kernelRadius; i <= kernelRadius; ++i) {
        for (int j = -kernelRadius; j <= kernelRadius; ++j) {
          // Traverse within the kernel
          const int dest_x = x + i, dest_y = y + j;
          if (dest_x < 0 || dest_x >= width || dest_y < 0 || dest_y >= height) continue;

          const auto &X = m_accColor(dest_x, dest_y);
          E += X;
          E2 += X * X;
          ++cnt;
        }
      }

      constexpr float k = 0.1f;
      E /= float(cnt);
      E2 /= float(cnt);
      Var = E2 - E * E;

      color = Clamp(color, E - Var * k, E + Var * k);

      float alpha = 0.9;
      if (!m_valid(x, y)) alpha = 1.0f;
      m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
    }
  }

  std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
  int height = frameInfo.m_beauty.m_height;
  int width = frameInfo.m_beauty.m_width;
  Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);

  constexpr int kernelRadius = 16;
#pragma omp parallel for default(none) shared(width, height, frameInfo, filteredImage)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float kernelSum = 0.0f;
      Float3 result;
      const auto &C_i = frameInfo.m_beauty(x, y);
      const auto &N_i = frameInfo.m_normal(x, y);
      const auto &P_i = frameInfo.m_position(x, y);

      for (int i = -kernelRadius; i <= kernelRadius; ++i) {
        for (int j = -kernelRadius; j <= kernelRadius; ++j) {
          // Traverse within the kernel
          const int dest_x = x + i, dest_y = y + j;
          if (dest_x < 0 || dest_x >= width || dest_y < 0 || dest_y >= height) continue;

          const auto &C_j = frameInfo.m_beauty(dest_x, dest_y);
          const auto &N_j = frameInfo.m_normal(dest_x, dest_y);
          const auto &P_j = frameInfo.m_position(dest_x, dest_y);

          float J = 0.0f;
          float dist = SqrDistance(P_i, P_j);
          if (i == 0 && j == 0) goto end;

          // Calculate kernel
          J += dist / (2 * m_sigmaCoord);
          J += SqrDistance(C_i, C_j) / (2 * m_sigmaColor);
          J += powf(SafeAcos(Dot(N_i, N_j)), 2) / (2 * m_sigmaNormal);
          if (dist > 0.0f)
            J += powf(Dot(N_i, (P_j - P_i) / sqrtf(dist)), 2) / (2 * m_sigmaPlane);

        end:
          J = expf(-J);
          kernelSum += J;
          result += frameInfo.m_beauty(dest_x, dest_y) * J;
        }
      }

      assert(kernelSum != 0.0f);
      filteredImage(x, y) = result / kernelSum;  // Energy preservation
    }
  }

  return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
  m_accColor.Copy(filteredColor);
  int height = m_accColor.m_height;
  int width = m_accColor.m_width;
  m_misc = CreateBuffer2D<Float3>(width, height);
  m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
  // Filter current frame
  Buffer2D<Float3> filteredColor;
  filteredColor = Filter(frameInfo);

  // Reproject previous frame color to current
  // m_useTemportal = false;
  if (m_useTemportal) {
    Reprojection(frameInfo);
    TemporalAccumulation(filteredColor);
  } else {
    Init(frameInfo, filteredColor);
  }

  // Maintain
  Maintain(frameInfo);
  if (!m_useTemportal) m_useTemportal = true;

  return m_accColor;
}
