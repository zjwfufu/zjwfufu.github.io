import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@1.2.9";

const canvas = document.getElementById("canvas");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
const modelDropdown = document.getElementById("modelDropdown");

const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, canvas);

// 只在 canvas 可见时渲染，节省 GPU
let isVisible = false;
const observer = new IntersectionObserver(
    ([entry]) => { isVisible = entry.isIntersecting; },
    { threshold: 0.1 }
);
observer.observe(canvas);

function showModal() {
    progressDialog.style.display = "block";
}

function closeModal() {
    progressDialog.style.display = "none";
    progressIndicator.value = 0;
}

async function loadModel(url) {
    showModal();
    await SPLAT.Loader.LoadAsync(url, scene, (progress) => {
        progressIndicator.value = progress * 100;
    });
    closeModal();
}

async function main() {
    // 加载下拉框中的第一个模型
    await loadModel(modelDropdown.options[0].value);

    const handleResize = () => {
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    };

    const frame = () => {
        if (isVisible) {
            controls.update();
            renderer.render(scene, camera);
        }
        requestAnimationFrame(frame);
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // 切换模型
    modelDropdown.addEventListener("change", async (event) => {
        scene.reset();
        await loadModel(event.target.value);
    });

    requestAnimationFrame(frame);
}

main();
