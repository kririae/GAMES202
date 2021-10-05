class PRTMaterial extends Material {
  constructor(vertexShader, fragmentShader) {
    super(
      {
        // Migrate into MeshRender.js func bindMaterialParameters()
        uPrecomputeLR: { type: "precomputeLR", value: null },
        uPrecomputeLG: { type: "precomputeLG", value: null },
        uPrecomputeLB: { type: "precomputeLB", value: null },
      },
      ["aPrecomputeLT"],
      vertexShader,
      fragmentShader,
      null
    );
  }
}

async function buildPRTMaterial(vertexPath, fragmentPath) {
  let vertexShader = await getShaderString(vertexPath);
  let fragmentShader = await getShaderString(fragmentPath);

  return new PRTMaterial(vertexShader, fragmentShader);
}
