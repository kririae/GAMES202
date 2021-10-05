#include <filesystem/resolver.h>
#include <nori/integrator.h>
#include <nori/ray.h>
#include <nori/scene.h>
#include <sh/default_image.h>
#include <sh/spherical_harmonics.h>
#include <stb_image.h>

#include <Eigen/Core>
#include <fstream>
#include <random>

NORI_NAMESPACE_BEGIN

namespace ProjEnv {
std::vector<std::unique_ptr<float[]>> LoadCubemapImages(
    const std::string &cubemapDir, int &width, int &height, int &channel) {
  std::vector<std::string> cubemapNames{"negx.jpg", "posx.jpg", "posy.jpg",
                                        "negy.jpg", "posz.jpg", "negz.jpg"};

  std::vector<std::unique_ptr<float[]>> images(6);

  for (int i = 0; i < 6; i++) {
    std::string filename = cubemapDir + "/" + cubemapNames[i];

    int w, h, c;
    float *image = stbi_loadf(filename.c_str(), &w, &h, &c, 3);

    if (!image) {
      std::cout << "Failed to load image: " << filename << std::endl;
      exit(-1);
    }

    if (i == 0) {
      // Set information
      width = w;
      height = h;
      channel = c;
    } else if (w != width || h != height || c != channel) {
      std::cout << "Dismatch resolution for 6 images in cubemap" << std::endl;
      exit(-1);
    }

    images[i] = std::unique_ptr<float[]>(image);
    // int index = (0 * 128 + 0) * channel;
    // std::cout << images[i][index + 0] << "\t" << images[i][index + 1] << "\t"
    //           << images[i][index + 2] << std::endl;
  }

  return images;  // return the vector of unique_ptr, without any modification
}

const Eigen::Vector3f cubemapFaceDirections[6][3] = {
    {{0, 0, 1}, {0, -1, 0}, {-1, 0, 0}},   // negx
    {{0, 0, 1}, {0, -1, 0}, {1, 0, 0}},    // posx
    {{1, 0, 0}, {0, 0, -1}, {0, -1, 0}},   // negy
    {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},     // posy
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},  // negz
    {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},    // posz
};

float CalcPreArea(const float &x, const float &y) {
  return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0));
}

float CalcArea(const float &u_, const float &v_, const int &width,
               const int &height) {
  // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
  // (0.5 is for texel center addressing)
  float u = (2.0 * (u_ + 0.5) / width) - 1.0;
  float v = (2.0 * (v_ + 0.5) / height) - 1.0;

  // shift from a demi texel, mean 1.0 / size  with u and v in [-1..1]
  float invResolutionW = 1.0 / width;
  float invResolutionH = 1.0 / height;

  // u and v are the -1..1 texture coordinate on the current face.
  // get projected area for this texel
  float x0 = u - invResolutionW;
  float y0 = v - invResolutionH;
  float x1 = u + invResolutionW;
  float y1 = v + invResolutionH;
  float angle = CalcPreArea(x0, y0) - CalcPreArea(x0, y1) -
                CalcPreArea(x1, y0) + CalcPreArea(x1, y1);

  return angle;
}

// template <typename T> T ProjectSH() {}
template <size_t SHOrder>
Eigen::Array3f SHIntegrator(const std::vector<Eigen::Array3f> &SHCoeffiecents,
                            const Vector3f &dir) {
  constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1);
  Eigen::Array3f res(0.0);
  for (int l = 0;; ++l) {
    for (int m = -l; m <= l; ++m) {
      int shIndex = l * (l + 1) + m;
      if (shIndex >= SHNum) goto _fin;
      res += SHCoeffiecents[shIndex] * sh::EvalSH(l, m, dir.cast<double>());
    }
  }
_fin:
  return res;
}

template <size_t SHOrder>
std::vector<Eigen::Array3f> PrecomputeCubemapSH(
    const std::vector<std::unique_ptr<float[]>> &images, const int &width,
    const int &height, const int &channel) {
  std::vector<Eigen::Vector3f> cubemapDirs;
  cubemapDirs.reserve(6 * width * height);

  // Calculate cubemapDirs
  for (int i = 0; i < 6; i++) {
    Eigen::Vector3f faceDirX = cubemapFaceDirections[i][0];
    Eigen::Vector3f faceDirY = cubemapFaceDirections[i][1];
    Eigen::Vector3f faceDirZ = cubemapFaceDirections[i][2];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float u = 2 * ((x + 0.5) / width) - 1;
        float v = 2 * ((y + 0.5) / height) - 1;
        Eigen::Vector3f dir =
            (faceDirX * u + faceDirY * v + faceDirZ).normalized();
        cubemapDirs.push_back(dir);
      }
    }
  }

  // ------------------------------------------------

  constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1);
  std::vector<Eigen::Array3f> SHCoeffiecents(SHNum);

  for (int i = 0; i < SHNum; i++)
    SHCoeffiecents[i] = Eigen::Array3f(0);  // For three channels

  // Monte Carlo Integration
  // float sumWeight = 0;

  for (int i = 0; i < 6; i++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // TODO: here you need to compute light sh of each face of cubemap of
        // each pixel
        Eigen::Vector3f dir = cubemapDirs[i * width * height + y * width + x];
        int index = (y * width + x) * channel;
        Eigen::Array3f Le(images[i][index + 0], images[i][index + 1],
                          images[i][index + 2]);

        // Le: three channels in image for the corresponding place
        // Calculate \varphi, \theta
        // float theta = std::acos(dir.z());
        // float phi = 0.0;
        // if (dir.x != 0)
        //   phi = std::atan(dir.y() / dir.x());  // Yet another calculation
        //   method
        // else                               // y = sign(y)
        //   phi = dir.y() * M_PI / 2.0;
        // assert(!std::isnan(phi));  // Correction

        // DEBUG
        // SHCoeffiecents[0] += Le * CalcArea(x, y, width, height) / 4.0 /
        // M_PI; goto _fin;  // skip

        for (int l = 0;; ++l) {
          for (int m = -l; m <= l; ++m) {
            int shIndex = l * (l + 1) + m;
            if (shIndex >= SHNum) goto _fin;
            SHCoeffiecents[shIndex] +=
                CalcArea(x, y, width, height) * Le *
                float(sh::EvalSH(l, m, dir.cast<double>()));
            // SHCoeffiecents[shIndex] += Le * CalcArea(x, y, width, height);
          }
        }
      _fin:
        continue;
      }
    }
  }
  return SHCoeffiecents;
}
}  // namespace ProjEnv

class PRTIntegrator : public Integrator {
 public:
  static constexpr int SHOrder = 2;
  static constexpr int SHCoeffLength = (SHOrder + 1) * (SHOrder + 1);

  enum class Type { Unshadowed = 0, Shadowed = 1, Interreflection = 2 };

  PRTIntegrator(const PropertyList &props) {
    /* No parameters this time */
    m_SampleCount = props.getInteger("PRTSampleCount", 100);
    m_CubemapPath = props.getString("cubemap");
    auto type = props.getString("type", "unshadowed");
    if (type == "unshadowed") {
      m_Type = Type::Unshadowed;
    } else if (type == "shadowed") {
      m_Type = Type::Shadowed;
    } else if (type == "interreflection") {
      m_Type = Type::Interreflection;
      m_Bounce = props.getInteger("bounce", 1);
    } else {
      throw NoriException("Unsupported type: %s.", type);
    }
  }

  virtual void preprocess(const Scene *scene) override {
    // Here only compute one mesh
    const auto mesh = scene->getMeshes()[0];

    // Projection environment
    auto cubePath = getFileResolver()->resolve(m_CubemapPath);
    auto lightPath = cubePath / "light.txt";
    auto transPath = cubePath / "transport.txt";
    std::ofstream lightFout(lightPath.str());
    std::ofstream fout(transPath.str());

    int width, height, channel;
    std::vector<std::unique_ptr<float[]>> images =
        ProjEnv::LoadCubemapImages(cubePath.str(), width, height, channel);

    auto envCoeffs =
        ProjEnv::PrecomputeCubemapSH<SHOrder>(images, width, height, channel);

    // Test Code, when we set Le as a constant
    // std::cout << ProjEnv::SHIntegrator<SHOrder>(envCoeffs,
    // Eigen::Vector3f(0, 0, 1))
    // << std::endl;

    m_LightCoeffs.resize(3, SHCoeffLength);
    for (int i = 0; i < envCoeffs.size(); i++) {
      lightFout << (envCoeffs)[i].x() << " " << (envCoeffs)[i].y() << " "
                << (envCoeffs)[i].z() << std::endl;
      m_LightCoeffs.col(i) = (envCoeffs)[i];
    }

    std::cout << "Computed light sh coeffs from: " << cubePath.str()
              << " to: " << lightPath.str() << std::endl;

    // Projection transport
    m_TransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount());
    fout << mesh->getVertexCount() << std::endl;

    for (int i = 0; i < mesh->getVertexCount(); i++) {
      const Point3f &v = mesh->getVertexPositions().col(i);
      const Normal3f &n = mesh->getVertexNormals().col(i);
      auto shFunc = [&](double phi, double theta) -> double {
        Eigen::Array3d d = sh::ToVector(phi, theta);
        const auto wi = Vector3f(d.x(), d.y(), d.z());
        if (m_Type == Type::Unshadowed) {
          return std::max(n.dot(wi), 0.0f);
        } else {
          return std::max(n.dot(wi), 0.0f) * !scene->rayIntersect(Ray3f(v, wi));
        }
      };

      auto shCoeff = sh::ProjectFunction(SHOrder, shFunc, m_SampleCount);

      for (int j = 0; j < shCoeff->size(); j++) {
        m_TransportSHCoeffs.col(i).coeffRef(j) = (*shCoeff)[j];
      }
    }

    if (m_Type == Type::Interreflection) {
      // TODO: leave for bonus
    }

    // Save in face format
    for (int f = 0; f < mesh->getTriangleCount(); f++) {
      const MatrixXu &F = mesh->getIndices();
      uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);
      for (int j = 0; j < SHCoeffLength; j++) {
        fout << m_TransportSHCoeffs.col(idx0).coeff(j) << " ";
      }
      fout << std::endl;

      for (int j = 0; j < SHCoeffLength; j++) {
        fout << m_TransportSHCoeffs.col(idx1).coeff(j) << " ";
      }
      fout << std::endl;

      for (int j = 0; j < SHCoeffLength; j++) {
        fout << m_TransportSHCoeffs.col(idx2).coeff(j) << " ";
      }
      fout << std::endl;
    }

    std::cout << "Computed SH coeffs"
              << " to: " << transPath.str() << std::endl;
  }

  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
    Intersection its;
    if (!scene->rayIntersect(ray, its)) return Color3f(0.0f);

    const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1>
        sh0 = m_TransportSHCoeffs.col(its.tri_index.x()),
        sh1 = m_TransportSHCoeffs.col(its.tri_index.y()),
        sh2 = m_TransportSHCoeffs.col(its.tri_index.z());
    const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1>
        rL = m_LightCoeffs.row(0),
        gL = m_LightCoeffs.row(1), bL = m_LightCoeffs.row(2);

    // m_LightCoeffs: n×3 matrix
    // 
    Color3f c0 = Color3f(rL.dot(sh0), gL.dot(sh0), bL.dot(sh0)),
            c1 = Color3f(rL.dot(sh1), gL.dot(sh1), bL.dot(sh1)),
            c2 = Color3f(rL.dot(sh2), gL.dot(sh2), bL.dot(sh2));

    const Vector3f &bary = its.bary;
    Color3f c = bary.x() * c0 + bary.y() * c1 + bary.z() * c2;

    // TODO: you need to delete the following four line codes after finishing
    // your calculation to SH,
    //       we use it to visualize the normals of model for debug.
    if (c.isZero()) {
      auto n_ = its.shFrame.n.cwiseAbs();
      return Color3f(n_.x(), n_.y(), n_.z());
    }
    return c;
  }

  std::string toString() const { return "PRTIntegrator[]"; }

 private:
  Type m_Type;
  int m_Bounce = 1;
  int m_SampleCount = 100;
  std::string m_CubemapPath;
  Eigen::MatrixXf m_TransportSHCoeffs;
  Eigen::MatrixXf m_LightCoeffs;
};

NORI_REGISTER_CLASS(PRTIntegrator, "prt");
NORI_NAMESPACE_END