import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';
import { ForwardPlusRenderer } from './renderers/forward_plus';
import { ClusteredDeferredRenderer } from './renderers/clustered_deferred';

import { setupLoaders, Scene } from './stage/scene';
import { Lights } from './stage/lights';
import { Volumetrics } from './stage/volumetrics';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();
const lights = new Lights(camera);
const volumetrics = new Volumetrics(camera, lights);

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();
gui.add(lights, 'numLights').min(1).max(Lights.maxNumLights).step(1).onChange(() => {
    lights.updateLightSetUniformNumLights();
    renderer?.resetTiming();
});

const stage = new Stage(scene, lights, volumetrics, camera, stats);

var renderer: Renderer | undefined;

function setRenderer(mode: string) {
    renderer?.stop();

    switch (mode) {
        case renderModes.naive:
            renderer = new NaiveRenderer(stage);
            break;
        case renderModes.forwardPlus:
            renderer = new ForwardPlusRenderer(stage);
            break;
        case renderModes.clusteredDeferred:
            renderer = new ClusteredDeferredRenderer(stage);
            break;
    }
}

const renderModes = { naive: 'naive', forwardPlus: 'forward+', clusteredDeferred: 'clustered deferred' };
let renderModeController = gui.add({ mode: renderModes.forwardPlus }, 'mode', renderModes);
renderModeController.onChange(setRenderer);

// Volumetric fog controls
const volumeUniforms = volumetrics.uniforms;
const resetTiming = () => renderer?.resetTiming();

const fogFolder = gui.addFolder('volumetric fog');
fogFolder.open();

const noiseFolder = fogFolder.addFolder('noise');
noiseFolder.open();
noiseFolder.add(volumeUniforms, 'densityScale', 0.0, 300.0, 1.0).name('density scale').onChange(resetTiming);
noiseFolder.add(volumeUniforms, 'noiseFreq', 0.05, 5.0, 0.01).name('frequency').onChange(resetTiming);
noiseFolder.add(volumeUniforms, 'octaveCount', 1, 8, 1).name('octaves').onChange(resetTiming);
noiseFolder.add(volumetrics, 'animationSpeed', 0.0, 5.0, 0.01).name('animation speed');

const mediaFolder = fogFolder.addFolder('media');
mediaFolder.open();
mediaFolder.add(volumeUniforms, 'scattering', 0.0, 0.2, 0.001).onChange(resetTiming);
mediaFolder.add(volumeUniforms, 'absorption', 0.0, 0.1, 0.001).onChange(resetTiming);
mediaFolder.add(volumeUniforms, 'phaseG', -0.95, 0.95, 0.01).name('phase g').onChange(resetTiming);
const [ambientR, ambientG, ambientB] = volumeUniforms.ambientLightColor;
const ambient = { r: ambientR, g: ambientG, b: ambientB };
const updateAmbient = () => {
    volumeUniforms.ambientLightColor = new Float32Array([ambient.r, ambient.g, ambient.b]);
};
mediaFolder.add(ambient, 'r', 0.0, 1.0, 0.01).name('ambient r').onChange(updateAmbient);
mediaFolder.add(ambient, 'g', 0.0, 1.0, 0.01).name('ambient g').onChange(updateAmbient);
mediaFolder.add(ambient, 'b', 0.0, 1.0, 0.01).name('ambient b').onChange(updateAmbient);

const gridFolder = fogFolder.addFolder('froxel grid');
gridFolder.add(volumeUniforms, 'volumeNearZ', 0.1, 5.0, 0.01).name('near z').onChange(resetTiming);
gridFolder.add(volumeUniforms, 'volumeFarZ', 10.0, 100.0, 1.0).name('far z').onChange(resetTiming);
gridFolder.add(volumeUniforms, 'distributionExponent', 0.25, 3.0, 0.01).name('slice exponent').onChange(resetTiming);

setRenderer(renderModeController.getValue());
