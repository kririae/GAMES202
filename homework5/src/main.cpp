#include <cassert>
#include <fstream>
#include <string>

#include "denoiser.h"
#include "util/image.h"
#include "util/mathutil.h"

std::vector<Matrix4x4> ReadMatrix(const std::string &filename) {
  std::ifstream is;
  is.open(filename, std::ios::binary);
  CHECK(is.is_open());
  int shapeNum;
  is.read(reinterpret_cast<char *>(&shapeNum), sizeof(int));
  std::vector<Matrix4x4> matrix(shapeNum + 2);
  for (int i = 0; i < shapeNum + 2; i++)
    is.read(reinterpret_cast<char *>(&matrix[i]), sizeof(Matrix4x4));
  is.close();
  return matrix;
}

FrameInfo LoadFrameInfo(const filesystem::path &inputDir, const int &idx) {
  Buffer2D<Float3> beauty =
      ReadFloat3Image((inputDir / ("beauty_" + std::to_string(idx) + ".exr")).str());
  Buffer2D<Float3> normal =
      ReadFloat3Image((inputDir / ("normal_" + std::to_string(idx) + ".exr")).str());
  Buffer2D<Float3> position =
      ReadFloat3Image((inputDir / ("position_" + std::to_string(idx) + ".exr")).str());
  Buffer2D<float> depth =
      ReadFloatImage((inputDir / ("depth_" + std::to_string(idx) + ".exr")).str());
  Buffer2D<float> id =
      ReadFloatImage((inputDir / ("ID_" + std::to_string(idx) + ".exr")).str());
  std::vector<Matrix4x4> matrix =
      ReadMatrix((inputDir / ("matrix_" + std::to_string(idx) + ".mat")).str());

  FrameInfo frameInfo = {beauty, depth, normal, position, id, matrix};
  return frameInfo;
}

void Denoise(const filesystem::path &inputDir, const filesystem::path &outputDir,
             const int &frameNum) {
  Denoiser denoiser;
  for (int i = 0; i < frameNum; i++) {
    std::cout << "Frame: " << i << std::endl;
    FrameInfo frameInfo = LoadFrameInfo(inputDir, i);
    Buffer2D<Float3> image = denoiser.ProcessFrame(frameInfo);
    std::string filename = (outputDir / ("result_" + std::to_string(i) + ".exr")).str();
    WriteFloat3Image(image, filename);
  }
}

std::string getPath(const std::string &target, int depth) {
  std::string path = target;
  for (int i = 0; i < depth; ++i) {
    FILE *file = fopen(path.c_str(), "r");
    if (file) {
      fclose(file);
      return path;
    }

    path = "../" + path;
  }

  LOG("failed to get file: " + target);
  return target;
}
int main() {
  // Box
  // filesystem::path inputDir(getPath("examples/box/input", 5));
  // filesystem::path outputDir(getPath("examples/box/output", 5));
  // int frameNum = 20;

  // Pink room
  filesystem::path inputDir(getPath("examples/pink-room/input", 5));
  filesystem::path outputDir(getPath("examples/pink-room/output", 5));
  int frameNum = 80;

  Denoise(inputDir, outputDir, frameNum);
  return 0;
}