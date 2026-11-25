import { device } from "../renderer";

import * as shaders from '../shaders/shaders';
import { Camera } from "./camera";
import { Lights } from "./lights";

class VolumeUniforms {
    static readonly bufferBytes = 16 * Float32Array.BYTES_PER_ELEMENT;
    readonly buffer = new ArrayBuffer(VolumeUniforms.bufferBytes);
    private readonly floatView = new Float32Array(this.buffer);
    private readonly uintView = new Uint32Array(this.buffer);

    get volumeNearZ(): number {
        return this.floatView[0];
    }

    set volumeNearZ(nearZ: number) {
        this.floatView.set(new Float32Array([nearZ]), 0);
    }

    get volumeFarZ(): number {
        return this.floatView[1];
    }

    set volumeFarZ(farZ: number) {
        this.floatView.set(new Float32Array([farZ]), 1);
    }

    get distributionExponent(): number {
        return this.floatView[2];
    }

    set distributionExponent(distrib: number) {
        this.floatView.set(new Float32Array([distrib]), 2);
    }

    get scattering(): number {
        return this.floatView[3];
    }

    set scattering(scattering: number) {
        this.floatView.set(new Float32Array([scattering]), 3);
    }

    get ambientLightColor(): Float32Array {
        return this.floatView.slice(4, 7);
    }

    set ambientLightColor(ambient: Float32Array) {
        this.floatView.set(ambient, 4);
    }

    get absorption(): number {
        return this.floatView[7];
    }

    set absorption(absorption: number) {
        this.floatView.set(new Float32Array([absorption]), 7);
    }

    get noiseFreq(): number {
        return this.floatView[8];
    }

    set noiseFreq(noiseFreq: number) {
        this.floatView.set(new Float32Array([noiseFreq]), 8);
    }

    get octaveCount(): number {
        return this.uintView[9];
    }

    set octaveCount(octaveCount: number) {
        this.uintView.set(new Uint32Array([octaveCount]), 9);
    }

    get densityScale(): number {
        return this.floatView[10];
    }

    set densityScale(densityScale: number) {
        this.floatView.set(new Float32Array([densityScale]), 10);
    }

    get phaseG(): number {
        return this.floatView[11];
    }

    set phaseG(g: number) {
        this.floatView.set(new Float32Array([g]), 11);
    }

    set sampleJitter(jitter: number) {
        this.floatView.set(new Float32Array([jitter]), 12);
    }

    set time(time: number) {
        this.floatView.set(new Float32Array([time]), 13);
    }

    get densityLow(): number {
        return this.floatView[14];
    }

    set densityLow(low: number) {
        this.floatView.set(new Float32Array([low]), 14);
    }

    get densityHigh(): number {
        return this.floatView[15];
    }

    set densityHigh(high: number) {
        this.floatView.set(new Float32Array([high]), 15);
    }
}

export class Volumetrics {
    private camera: Camera;
    private lights: Lights;

    uniforms: VolumeUniforms = new VolumeUniforms();
    uniformsBuffer: GPUBuffer;

    frameCount: number = 0;

    // Multiplier on how quickly the noise animates (0 stops the animation)
    animationSpeed: number = 0.5;
    private animationTime: number = 0;
    private lastFrameTimeMs: number = performance.now();

    // A 3D texture used to store a volume's scattering and extinction
    // coefficients. One texel represents the data for one froxel.
    scatteringExtinctionBuffer: GPUTexture;
    scatteringExtinctionBufferView: GPUTextureView;

    lightingIndex: number = 0;
    lightingBuffers: GPUTexture[] = [];
    lightingBufferViews: GPUTextureView[] = [];

    integrationBuffer: GPUTexture;
    public integrationBufferView: GPUTextureView;

    // The texture sampler we'll use to sample our volume buffers
    // during the integration/composition pass
    volumeSampler: GPUSampler;

    voxelizationComputeBindGroupLayout: GPUBindGroupLayout;
    voxelizationComputeBindGroup: GPUBindGroup;
    voxelizationComputePipeline: GPUComputePipeline;

    lightingComputeBindGroupLayout: GPUBindGroupLayout;
    lightingComputeBindGroups: GPUBindGroup[] = [];
    lightingComputePipeline: GPUComputePipeline;

    integrationComputeBindGroupLayout: GPUBindGroupLayout;
    integrationComputeBindGroups: GPUBindGroup[] = [];
    integrationComputePipeline: GPUComputePipeline;

    constructor(camera: Camera, lights: Lights) {
        this.camera = camera;
        this.lights = lights;

        // Create all the buffers/textures we need
        this.uniformsBuffer = device.createBuffer({
            label: "volumetricsUniforms",
            size: this.uniforms.buffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.scatteringExtinctionBuffer = device.createTexture({
            label: "scatteringExtinctionBuffer",
            dimension: "3d",
            size: [shaders.constants.froxelGridDimX,
                shaders.constants.froxelGridDimY,
                shaders.constants.froxelGridDimZ],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        this.scatteringExtinctionBufferView = this.scatteringExtinctionBuffer.createView();

        for (let i = 0; i < 2; i++) {
            this.lightingBuffers[i] = device.createTexture({
                label: `lightingBuffer_${i}`,
                dimension: "3d",
                size: [shaders.constants.froxelGridDimX,
                shaders.constants.froxelGridDimY,
                shaders.constants.froxelGridDimZ],
                format: "rgba16float",
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
            });

            this.lightingBufferViews[i] = this.lightingBuffers[i].createView();
        }

        this.integrationBuffer = device.createTexture({
            label: "integrationBuffer",
            dimension: "3d",
            size: [shaders.constants.froxelGridDimX,
            shaders.constants.froxelGridDimY,
            shaders.constants.froxelGridDimZ],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        this.integrationBufferView = this.integrationBuffer.createView();

        this.volumeSampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            addressModeW: "clamp-to-edge"
        });

        // Create the bind group, bind group layout, and pipeline for
        // our voxelization compute pass
        this.voxelizationComputeBindGroupLayout = device.createBindGroupLayout({
            label: "voxelization compute bind group layout",
            entries: [
                { // camera
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // volumeUniforms
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // scatteringExtinctionBuffer
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" }
                }
            ]
        });

        this.voxelizationComputeBindGroup = device.createBindGroup({
            label: "voxelization compute bind group",
            layout: this.voxelizationComputeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.camera.uniformsBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.uniformsBuffer }
                },
                {
                    binding: 2,
                    resource: this.scatteringExtinctionBufferView
                }
            ]
        });

        this.voxelizationComputePipeline = device.createComputePipeline({
            label: "voxelization compute pipeline",
            layout: device.createPipelineLayout({
                label: "voxelization compute pipeline layout",
                bindGroupLayouts: [this.voxelizationComputeBindGroupLayout]
            }),
            compute: {
                module: device.createShaderModule({
                    label: "voxelization compute shader",
                    code: shaders.volumetricVoxelizationComputeSrc
                }),
                entryPoint: "populateParticipatingMediaProperties"
            }
        });

        // Create the bind group, bind group layout, and pipeline for
        // our volumetric lighting compute pass
        this.lightingComputeBindGroupLayout = device.createBindGroupLayout({
            label: "volumetric lighting compute bind group layout",
            entries: [
                { // volumeUniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // scatteringExtinctionBuffer
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float", viewDimension: "3d" }
                },
                { // lightingBuffer
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" }
                },
                { // camera
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // lightSet
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // clusterSet
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // lightingHistory
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float", viewDimension: "3d" }
                },
                { // volumeSampler
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: { type: "filtering" }
                }
            ]
        });

        for (let i = 0; i < 2; i++) {
            this.lightingComputeBindGroups[i] = device.createBindGroup({
                label: "volumetric lighting compute bind group",
                layout: this.lightingComputeBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformsBuffer }
                    },
                    {
                        binding: 1,
                        resource: this.scatteringExtinctionBufferView
                    },
                    {
                        binding: 2,
                        resource: this.lightingBufferViews[i]
                    },
                    {
                        binding: 3,
                        resource: { buffer: this.camera.uniformsBuffer }
                    },
                    {
                        binding: 4,
                        resource: { buffer: this.lights.lightSetStorageBuffer }
                    },
                    {
                        binding: 5,
                        resource: { buffer: this.lights.clusterSetStorageBuffer }
                    },
                    {
                        binding: 6,
                        resource: this.lightingBufferViews[1 - i]
                    },
                    {
                        binding: 7,
                        resource: this.volumeSampler
                    }
                ]
            });
        }

        this.lightingComputePipeline = device.createComputePipeline({
            label: "volumetric lighting compute pipeline",
            layout: device.createPipelineLayout({
                label: "volumetric lighting compute pipeline layout",
                bindGroupLayouts: [this.lightingComputeBindGroupLayout]
            }),
            compute: {
                module: device.createShaderModule({
                    label: "volumetric lighting compute shader",
                    code: shaders.volumetricLightingComputeSrc
                }),
                entryPoint: "computeScatteredLight"
            }
        });

        // Create the bind group, bind group layout, and pipeline for
        // our integration compute pass
        this.integrationComputeBindGroupLayout = device.createBindGroupLayout({
            label: "integration compute bind group layout",
            entries: [
                { // volumeUniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // lightingVolume
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float", viewDimension: "3d" }
                },
                { // integrationBuffer
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" }
                }
            ]
        });

        for (let i = 0; i < 2; i++) {
            this.integrationComputeBindGroups[i] = device.createBindGroup({
                label: "integration compute bind group",
                layout: this.integrationComputeBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformsBuffer }
                    },
                    {
                        binding: 1,
                        resource: this.lightingBufferViews[i]
                    },
                    {
                        binding: 2,
                        resource: this.integrationBufferView
                    }
                ]
            });
        }

        this.integrationComputePipeline = device.createComputePipeline({
            label: "integration compute pipeline",
            layout: device.createPipelineLayout({
                label: "integration compute pipeline layout",
                bindGroupLayouts: [this.integrationComputeBindGroupLayout]
            }),
            compute: {
                module: device.createShaderModule({
                    label: "integration compute shader",
                    code: shaders.volumetricIntegrationComputeSrc
                }),
                entryPoint: "integrateVolume"
            }
        });

        // Set our default volume uniform values and write them to the GPU
        // (Note that these are GUI-controllable)
        this.uniforms.volumeNearZ = 0.2;
        this.uniforms.volumeFarZ = 40.0;
        this.uniforms.distributionExponent = 1.0;
        this.uniforms.scattering = 0.04;
        this.uniforms.ambientLightColor = new Float32Array([0.15, 0.15, 0.15]);
        this.uniforms.absorption = 0.01;
        this.uniforms.noiseFreq = 10.0;
        this.uniforms.octaveCount = 4;
        this.uniforms.densityScale = 30.0;
        this.uniforms.phaseG = 0.3;
        this.uniforms.densityLow = 0.05;
        this.uniforms.densityHigh = 0.2;
    }

    computeVolumetrics(encoder: GPUCommandEncoder) {
        this.uniforms.sampleJitter = (this.frameCount * 0.61803398875) % 1;
        this.frameCount++;

        // Accumulate time so that changing the speed doesn't make the noise jump
        const nowMs = performance.now();
        this.animationTime += ((nowMs - this.lastFrameTimeMs) / 1000) * this.animationSpeed;
        this.lastFrameTimeMs = nowMs;
        this.uniforms.time = this.animationTime;

        device.queue.writeBuffer(this.uniformsBuffer, 0, this.uniforms.buffer);

        const workgroupCountX = Math.ceil(shaders.constants.froxelGridDimX / shaders.constants.froxelWorkgroupSizeX);
        const workgroupCountY = Math.ceil(shaders.constants.froxelGridDimY / shaders.constants.froxelWorkgroupSizeY);
        const workgroupCountZ = Math.ceil(shaders.constants.froxelGridDimZ / shaders.constants.froxelWorkgroupSizeZ);

        const computePass = encoder.beginComputePass();

        computePass.setPipeline(this.voxelizationComputePipeline);
        computePass.setBindGroup(shaders.constants.bindGroup_volumetric, this.voxelizationComputeBindGroup);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);

        computePass.setPipeline(this.lightingComputePipeline);
        computePass.setBindGroup(shaders.constants.bindGroup_volumetric, this.lightingComputeBindGroups[this.lightingIndex]);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);

        computePass.setPipeline(this.integrationComputePipeline);
        computePass.setBindGroup(shaders.constants.bindGroup_volumetric, this.integrationComputeBindGroups[this.lightingIndex]);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

        computePass.end();

        this.lightingIndex = 1 - this.lightingIndex;
    }
}
