attribute mat3 aPrecomputeLT;
attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform mat3 uPrecomputeLR;
uniform mat3 uPrecomputeLG;
uniform mat3 uPrecomputeLB;

varying highp vec3 vFragPos;
// varying highp vec3 vNormal;
varying highp vec3 vColor;

highp float dotMat(mat3 x, mat3 y) {
  vec3 x_col_0 = x[0], x_col_1 = x[1], x_col_2 = x[2];
  vec3 y_col_0 = y[0], y_col_1 = y[1], y_col_2 = y[2];
  return dot(x_col_0, y_col_0) + dot(x_col_1, y_col_1) + dot(x_col_2, y_col_2);
}

void main(void) {
  vColor = vec3(dotMat(uPrecomputeLR, aPrecomputeLT),
                dotMat(uPrecomputeLG, aPrecomputeLT),
                dotMat(uPrecomputeLB, aPrecomputeLT));
  vFragPos = (uModelMatrix * vec4(aVertexPosition, 1.0)).xyz;
  gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix *
                vec4(aVertexPosition, 1.0);
}