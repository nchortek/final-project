@group(${bindGroup_volumetric}) @binding(0) var<uniform> volume: VolumeUniforms;
@group(${bindGroup_volumetric}) @binding(1) var scatteringExtinctionBuffer: texture_3d<f32>;
@group(${bindGroup_volumetric}) @binding(2) var lightingBuffer: texture_storage_3d<rgba16float, write>;
@group(${bindGroup_volumetric}) @binding(3) var<uniform> camera: CameraUniforms;
@group(${bindGroup_volumetric}) @binding(4) var<storage, read> lightSet: LightSet;
@group(${bindGroup_volumetric}) @binding(5) var<storage, read> clusterSet: ClusterSet;
@group(${bindGroup_volumetric}) @binding(6) var lightingHistory: texture_3d<f32>;
@group(${bindGroup_volumetric}) @binding(7) var volumeSampler: sampler;

const historyBlendWeightStill: f32 = 0.95;
const historyBlendWeightMoving: f32 = 0.7;
const motionSensitivity: f32 = 8.0;

// Henyey-Greenstein phase function -- calculates the proportion of light that
// a volume scatters towards the camera
fn phase(dirToLight: vec3f, dirToCamera: vec3f, g: f32) -> f32 {
    let cosTheta = dot(dirToLight, dirToCamera);
    var denom: f32 = 1.0 + (g * g) + (2.0 * g * cosTheta);
    denom = max(denom, 1e-6);
    let inv4pi: f32 = 1.0 / (4.0 * 3.1415926);
    return inv4pi * (1.0 - (g * g)) / (denom * sqrt(denom));
}

@compute
@workgroup_size(${froxelWorkgroupSizeX}, ${froxelWorkgroupSizeY}, ${froxelWorkgroupSizeZ})
fn computeScatteredLight(@builtin(global_invocation_id) id: vec3u) {
    if (any(id >= textureDimensions(lightingBuffer))) {
        return;
    }

    let participatingMediaProperties: vec4f = textureLoad(scatteringExtinctionBuffer, id, 0);
    var totalLightContrib = vec3f(0.0);

    let froxelGridDims: vec3f = vec3f(f32(${froxelGridDimX}), f32(${froxelGridDimY}), f32(${froxelGridDimZ}));

    // We add 0.5 to both our uv and froxel slice computations because we want
    // to obtain the world position of the center of the froxel
    let uv: vec2f = (vec2f(id.xy) + 0.5) / froxelGridDims.xy;
    let froxelZSlice: f32 = f32(id.z) + 0.5;
    let froxelWorldPos: vec3f = getFroxelWorldPosition(
        uv,
        froxelZSlice,
        froxelGridDims.z,
        volume.volumeNearZ,
        volume.volumeFarZ,
        volume.distributionExponent,
        camera.invProjMat,
        camera.invViewMat);

    let jitteredZSlice: f32 = f32(id.z) + volume.sampleJitter;
    let jitteredWorldPos: vec3f = getFroxelWorldPosition(
        uv,
        jitteredZSlice,
        froxelGridDims.z,
        volume.volumeNearZ,
        volume.volumeFarZ,
        volume.distributionExponent,
        camera.invProjMat,
        camera.invViewMat);

    // Froxels without media have nothing to scatter, so skip the lighting work.
    // They still go through the temporal blend below as a sample of zero.
    if (participatingMediaProperties.a > 0.0) {
        let viewZ: f32 = getViewZFromFroxelZSlice(
            jitteredZSlice,
            froxelGridDims.z,
            volume.volumeNearZ,
            volume.volumeFarZ,
            volume.distributionExponent);

        let flatClusterIdx: u32 = getClusterIndex(viewZ, uv);
        let cameraWorldPos: vec3f = camera.invViewMat[3].xyz;
        let vecToCamera = normalize(cameraWorldPos - jitteredWorldPos);

        for (var lightIdx = 0u; lightIdx < clusterSet.clusters[flatClusterIdx].numLights; lightIdx++) {
            let light = lightSet.lights[clusterSet.clusters[flatClusterIdx].lightIndices[lightIdx]];

            // Calculate light contribution
            let vecToLight = light.pos - jitteredWorldPos;
            let distToLight = length(vecToLight);
            let attenuation: f32 = rangeAttenuation(distToLight);
            totalLightContrib += (light.color * attenuation * phase(normalize(vecToLight), vecToCamera, volume.phaseG));
        }

        totalLightContrib += volume.ambientLightColor;
    }

    // Write the radiance scattered toward the camera (plus the extinction coefficient) for this froxel,
    // after blending with the previous frame's results
    var result: vec4f = vec4f(
        totalLightContrib * participatingMediaProperties.rgb,
        max(participatingMediaProperties.a, 0.0));

    let prevClip: vec4f = camera.prevViewProjMat * vec4f(froxelWorldPos, 1.0);
    let prevViewZ: f32 = prevClip.w;

    // Was the current froxel's world position at a depth where we store volumetric data
    // in the previous frame?
    if (prevViewZ >= volume.volumeNearZ
        && prevViewZ <= volume.volumeFarZ) {

        let prevNDC: vec2f = prevClip.xy / prevViewZ;
        let prevUV: vec2f = vec2f((prevNDC.x * 0.5) + 0.5, (-prevNDC.y * 0.5) + 0.5);

        // Did the current froxel's world position project onto the screen in the previous frame?
        if (all(prevUV >= vec2f(0.0))
            && all(prevUV <= vec2f(1.0))) {

            let prevFroxelSlice: f32 = getContinuousFroxelZSlice(
                froxelGridDims.z,
                prevViewZ,
                volume.volumeNearZ,
                volume.volumeFarZ,
                volume.distributionExponent);
            let prevUVW: vec3f = vec3f(prevUV, prevFroxelSlice / froxelGridDims.z);

            let history: vec4f = textureSampleLevel(lightingHistory, volumeSampler, prevUVW, 0.0);
            let motionBlendWeight: f32 = clamp(length(prevUV - uv) * motionSensitivity, 0.0, 1.0);
            let historyBlendWeight: f32 = mix(historyBlendWeightStill, historyBlendWeightMoving, motionBlendWeight);
            result = mix(result, history, historyBlendWeight);
        }
    }

    textureStore(lightingBuffer, id, result);
}