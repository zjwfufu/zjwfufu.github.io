document.addEventListener("DOMContentLoaded", function() {
    // 1. Initialize Three.js Scene
    const container = document.getElementById('starmap-container');
    if (!container) return;

    const scene = new THREE.Scene();

    // Create a beautiful deep space background gradient
    // We'll use a large sphere with a gradient texture as the background
    function createBackgroundSphere() {
        // Simple dark blue background instead of complex gradient sphere
        scene.background = new THREE.Color(0x020813);
    }
    createBackgroundSphere();

    // Add ambient galactic glow in the center (Removed as requested)
    // Removed the "dirty" looking center glow sphere

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 60, 120);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Setup OrbitControls for WASD/Mouse drag
    let controls;
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true; // Auto rotate slightly when not interacting
        controls.autoRotateSpeed = 0.5;
        controls.maxDistance = 300;
        controls.minDistance = 5;
    }

    // 2. Fetch Data
    fetch('/content.json')
        .then(response => response.json())
        .then(data => {
            const posts = data.posts || data; // Handle depending on JSON structure
            createGalaxy(posts);
        })
        .catch(err => console.error("Error loading posts for star map:", err));

    let points, postsData = [];
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    // UI Elements
    const infoPanel = document.getElementById('starmap-info-panel');
    const infoTitle = document.getElementById('starmap-info-title');
    const infoMeta = document.getElementById('starmap-info-meta');
    const infoExcerpt = document.getElementById('starmap-info-excerpt');
    const infoLink = document.getElementById('starmap-info-link');

    // Selection Box Elements (Tech style)
    let selectionGroup = new THREE.Group();
    scene.add(selectionGroup);

    function createSelectionBox() {
        const material = new THREE.LineBasicMaterial({ color: 0x66CCFF, linewidth: 2 });
        const size = 3;
        const length = 1;

        // Create 4 corners
        const points = [
            // Top Left
            [-size, size - length, 0], [-size, size, 0], [-size + length, size, 0],
            // Top Right
            [size - length, size, 0], [size, size, 0], [size, size - length, 0],
            // Bottom Right
            [size, -size + length, 0], [size, -size, 0], [size - length, -size, 0],
            // Bottom Left
            [-size + length, -size, 0], [-size, -size, 0], [-size, -size + length, 0]
        ];

        for (let i = 0; i < 4; i++) {
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...points[i * 3]),
                new THREE.Vector3(...points[i * 3 + 1]),
                new THREE.Vector3(...points[i * 3 + 2])
            ]);
            const line = new THREE.Line(geometry, material);
            selectionGroup.add(line);
        }
        selectionGroup.visible = false;
    }
    createSelectionBox();

    // Lines to nearest neighbors
    let connectionsGroup = new THREE.Group();
    scene.add(connectionsGroup);


    // Create a circular texture for stars
    function createCircleTexture(isCore = false) {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const context = canvas.getContext('2d');

        const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);

        if (isCore) {
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.1, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(0.4, 'rgba(255,255,255,0.2)');
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
        } else {
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.2, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.5, 'rgba(255,255,255,0.5)');
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
        }

        context.fillStyle = gradient;
        context.fillRect(0, 0, 64, 64);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    let initialPositions = null; // Store initial star positions to reset them

    // 3. Create Galaxy
    function createGalaxy(posts) {
        postsData = posts;
        const particleCount = posts.length;

        // Create main interactive stars
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        initialPositions = new Float32Array(particleCount * 3); // Allocate initial
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        // Map categories to specific branches
        const categoryMap = {};
        let branchCount = 0;

        posts.forEach(post => {
            let catName = 'Uncategorized';
            if (post.categories && post.categories.length > 0) {
                catName = post.categories[0].name;
            }
            if (categoryMap[catName] === undefined) {
                categoryMap[catName] = branchCount++;
            }
        });

        const totalBranches = Math.max(4, branchCount);
        const radius = 90; // Slightly larger galaxy
        const spin = 1.5;
        const randomness = 15;
        const randomnessPower = 2.5;

        // Unified Blue color palette with slight variations
        const baseBlue = new THREE.Color(0x33BBFF); // Brighter, more vibrant blue

        // Find the maximum word count to normalize star sizes
        let maxWords = 1;
        posts.forEach(post => {
            const wordCount = post.text ? post.text.length : (post.excerpt ? post.excerpt.length : 100);
            if (wordCount > maxWords) maxWords = wordCount;
        });

        for (let i = 0; i < particleCount; i++) {
            const post = posts[i];

            let catName = 'Uncategorized';
            if (post.categories && post.categories.length > 0) {
                catName = post.categories[0].name;
            }
            const branchIndex = categoryMap[catName];

            // Position: denser in center, sparse at edges, but push outward slightly
            const radiusDistance = Math.pow(Math.random(), 1.2) * radius + 15; // +15 keeps them away from absolute center

            const branchAngle = ((branchIndex % totalBranches) / totalBranches) * Math.PI * 2;
            const spinAngle = radiusDistance * spin / radius;

            const randomX = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;
            const randomY = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;
            const randomZ = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;

            positions[i * 3]     = Math.cos(branchAngle + spinAngle) * radiusDistance + randomX;
            positions[i * 3 + 1] = randomY * 0.3; // Flat disc
            positions[i * 3 + 2] = Math.sin(branchAngle + spinAngle) * radiusDistance + randomZ;

            initialPositions[i * 3]     = positions[i * 3];
            initialPositions[i * 3 + 1] = positions[i * 3 + 1];
            initialPositions[i * 3 + 2] = positions[i * 3 + 2];

            const baseColor = baseBlue.clone();
            const hsl = {};
            baseColor.getHSL(hsl);

            // Center is brighter
            const lightness = Math.max(0.6, Math.min(1.0, 1.0 - (radiusDistance / radius) * 0.3 + (Math.random() - 0.5) * 0.1));

            // Very slight hue shifts (staying strictly in blue/cyan range)
            baseColor.setHSL(
                hsl.h + (Math.random() - 0.5) * 0.05,
                Math.max(0.8, Math.min(1.0, hsl.s + (Math.random() - 0.5) * 0.1)),
                lightness
            );

            colors[i * 3]     = baseColor.r;
            colors[i * 3 + 1] = baseColor.g;
            colors[i * 3 + 2] = baseColor.b;

            // Size: proportional to word count
            const wordCount = post.text ? post.text.length : (post.excerpt ? post.excerpt.length : 100);
            // Size ranges from 6 to 18 based on length
            sizes[i] = 6 + (wordCount / maxWords) * 12;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1)); // We need to use a custom shader to use this size array

        // To use the individual sizes array, we need ShaderMaterial instead of PointsMaterial
        // Alternatively, if we just want basic scaling without rewriting shaders,
        // we can group them by size, or we inject the size attribute.
        // For simplicity and performance, we'll write a quick custom shader:
        const vertexShader = `
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;
        const fragmentShader = `
            uniform sampler2D pointTexture;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0) * texture2D(pointTexture, gl_PointCoord);
            }
        `;

        const material = new THREE.ShaderMaterial({
            uniforms: {
                pointTexture: { value: createCircleTexture() }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true,
            vertexColors: true
        });

        points = new THREE.Points(geometry, material);
        scene.add(points);

        // Add thousands of tiny background dust particles
        createCosmicDust();

        // Add the Black Hole at the center
        createBlackHole();

        animate();
    }

    let blackHoleMesh;

    function createBlackHole() {
        // Outer glowing accretion disk
        const accretionGeometry = new THREE.RingGeometry(4, 12, 64);
        const accretionMaterial = new THREE.MeshBasicMaterial({
            color: 0x33BBFF,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            map: createCircleTexture(true) // Reuse gradient texture to soften edges
        });
        const accretionDisk = new THREE.Mesh(accretionGeometry, accretionMaterial);
        accretionDisk.rotation.x = Math.PI / 2; // Lay flat

        // Dark sphere in center
        const bhGeometry = new THREE.SphereGeometry(3.8, 32, 32);
        const bhMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        const bhSphere = new THREE.Mesh(bhGeometry, bhMaterial);

        blackHoleMesh = new THREE.Group();
        blackHoleMesh.add(accretionDisk);
        blackHoleMesh.add(bhSphere);

        // Add an invisible slightly larger hit box for raycaster
        const hitGeometry = new THREE.SphereGeometry(6, 16, 16);
        const hitMaterial = new THREE.MeshBasicMaterial({ visible: false });
        const hitBox = new THREE.Mesh(hitGeometry, hitMaterial);
        hitBox.userData = { isBlackHole: true };
        blackHoleMesh.add(hitBox);

        scene.add(blackHoleMesh);
    }

    function createCosmicDust() {
        const dustCount = 4000; // slightly more dust
        const dustGeo = new THREE.BufferGeometry();
        const dustPos = new Float32Array(dustCount * 3);
        const dustCol = new Float32Array(dustCount * 3);

        const baseBlue = new THREE.Color(0x2288CC);

        for(let i=0; i<dustCount; i++) {
            // Spherical distribution concentrated in center
            const r = Math.pow(Math.random(), 2) * 150;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);

            // Flatten slightly
            dustPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
            dustPos[i*3+1] = r * Math.cos(phi) * 0.2; // very flat
            dustPos[i*3+2] = r * Math.sin(phi) * Math.sin(theta);

            // Keep dust strictly in blue spectrum
            const color = baseBlue.clone();
            const hsl = {};
            color.getHSL(hsl);
            const lightness = Math.random() * 0.3 + 0.1;
            color.setHSL(hsl.h + (Math.random() - 0.5) * 0.05, hsl.s, lightness);

            dustCol[i*3] = color.r;
            dustCol[i*3+1] = color.g;
            dustCol[i*3+2] = color.b;
        }

        dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
        dustGeo.setAttribute('color', new THREE.BufferAttribute(dustCol, 3));

        const dustMat = new THREE.PointsMaterial({
            size: 2.5, // Slightly larger dust
            sizeAttenuation: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            vertexColors: true,
            map: createCircleTexture(true),
            transparent: true,
            opacity: 0.6
        });

        const dustPoints = new THREE.Points(dustGeo, dustMat);
        scene.add(dustPoints);
    }

    // 4. Interactivity
    let hoveredIndex = null;
    let hoveredBlackHole = false;
    let selectedIndex = null;
    let targetCameraPos = new THREE.Vector3();
    let isZooming = false;
    let isWarping = false;

    // Remove old starmap-tooltip logic since we use the side panel now
    function onMouseMove(event) {
        if (container.style.display === 'none') return;
        if (isZooming || isWarping) return;

        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        // Check black hole first
        if (blackHoleMesh) {
            raycaster.setFromCamera(mouse, camera);
            const bhIntersects = raycaster.intersectObject(blackHoleMesh, true);
            if (bhIntersects.length > 0) {
                hoveredBlackHole = true;
                container.style.cursor = 'pointer';
                if (controls) controls.autoRotate = false;
                hoveredIndex = null;
                return;
            }
        }
        hoveredBlackHole = false;

        if (points) {
            raycaster.params.Points.threshold = 2.0; // Larger hit area
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);

            if (intersects.length > 0) {
                const intersect = intersects[0];
                if (hoveredIndex !== intersect.index) {
                    hoveredIndex = intersect.index;
                    container.style.cursor = 'pointer';
                    if (controls) controls.autoRotate = false;
                }
            } else {
                hoveredIndex = null;
                container.style.cursor = 'default';
                if (controls && selectedIndex === null && !hoveredBlackHole) controls.autoRotate = true;
            }
        }
    }

    function triggerWarp() {
        isWarping = true;
        if (controls) controls.enabled = false;
        hideInfoPanel();
        selectionGroup.visible = false;
        connectionsGroup.clear();

        // 1. Camera gets sucked in
        if (typeof TWEEN !== 'undefined') {
            new TWEEN.Tween(camera.position)
                .to({ x: 0, y: 5, z: 0 }, 2500)
                .easing(TWEEN.Easing.Cubic.In)
                .start();

            if (controls) {
                new TWEEN.Tween(controls.target)
                    .to({ x: 0, y: 0, z: 0 }, 1000)
                    .start();
            }
        }

        // 2. Stars get sucked in
        const positions = points.geometry.attributes.position.array;
        const particleCount = positions.length / 3;

        // Create tweens for every star (heavy, but cool)
        for (let i = 0; i < particleCount; i++) {
            const startPos = {
                x: positions[i * 3],
                y: positions[i * 3 + 1],
                z: positions[i * 3 + 2]
            };

            const delay = Math.random() * 1000;
            const duration = 1000 + Math.random() * 1000;

            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(startPos)
                    .to({ x: 0, y: 0, z: 0 }, duration)
                    .delay(delay)
                    .easing(TWEEN.Easing.Quadratic.In)
                    .onUpdate(() => {
                        positions[i * 3] = startPos.x;
                        positions[i * 3 + 1] = startPos.y;
                        positions[i * 3 + 2] = startPos.z;
                        points.geometry.attributes.position.needsUpdate = true;
                    })
                    .start();
            }
        }

        // 3. Black hole spins faster
        if (typeof TWEEN !== 'undefined') {
            const rotationTween = { speed: 0.05 };
            new TWEEN.Tween(rotationTween)
                .to({ speed: 0.5 }, 2000)
                .onUpdate(() => {
                    if (blackHoleMesh) {
                        blackHoleMesh.children[0].rotation.z += rotationTween.speed;
                    }
                })
                .start();
        }

        // End after animation
        setTimeout(() => {
            window.toggleStarMap();
        }, 2600);
    }

    function onClick(event) {
        if (container.style.display === 'none') return;
        if (isWarping) return;
        if (event.target && event.target.id === 'starmap-close') return;
        if (event.target && event.target.closest('#starmap-info-panel')) return; // Don't trigger if clicking inside panel

        if (hoveredBlackHole) {
            // Trigger the warp effect
            infoTitle.innerHTML = '？？？';
            infoMeta.innerHTML = '';
            infoExcerpt.innerHTML = '？？？';
            infoLink.innerHTML = '折跃 &rarr;';
            infoLink.href = 'javascript:void(0)';
            infoLink.onclick = function(e) {
                e.preventDefault();
                triggerWarp();
            };

            infoPanel.style.display = 'block';
            setTimeout(() => {
                infoPanel.style.transform = 'translateX(0)';
            }, 10);
            return;
        }

        if (hoveredIndex !== null && postsData[hoveredIndex]) {
            selectedIndex = hoveredIndex;
            const post = postsData[hoveredIndex];

            // 1. Show side panel with post info
            infoTitle.innerHTML = post.title || 'Untitled';

            // Format meta data
            let metaHtml = '';
            if (post.date) {
                const date = new Date(post.date);
                metaHtml += `<span>📅 ${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}</span>`;
            }
            if (post.categories && post.categories.length > 0) {
                metaHtml += `<span>📁 ${post.categories[0].name}</span>`;
            }
            infoMeta.innerHTML = metaHtml;

            // Format excerpt
            let excerptStr = post.excerpt || post.text || '浩瀚宇宙中，这颗星星上还没有留下摘录...';
            // Strip HTML tags roughly for display
            excerptStr = excerptStr.replace(/<[^>]+>/g, '').substring(0, 150) + '...';
            infoExcerpt.innerHTML = excerptStr;

            // Prepare link
            let rootPath = typeof yiliaConfig !== 'undefined' && yiliaConfig.root ? yiliaConfig.root : '/';
            const cleanPath = post.path.startsWith('/') ? post.path.substring(1) : post.path;
            const cleanRoot = rootPath.endsWith('/') ? rootPath : rootPath + '/';
            infoLink.innerHTML = '折跃 &rarr;';
            infoLink.href = cleanRoot + cleanPath;
            infoLink.onclick = null; // Clear warp click handler if any

            // Slide panel in
            infoPanel.style.display = 'block';
            setTimeout(() => {
                infoPanel.style.transform = 'translateX(0)';
            }, 10);

            // 2. Zoom camera to the star
            // Get star position
            const positions = points.geometry.attributes.position.array;
            const targetX = positions[selectedIndex * 3];
            const targetY = positions[selectedIndex * 3 + 1];
            const targetZ = positions[selectedIndex * 3 + 2];

            const starPos = new THREE.Vector3(targetX, targetY, targetZ);

            // Calculate a position slightly offset from the star to look at it
            // We want to move along the vector from center to star, slightly past it or above it
            const offsetDir = starPos.clone().normalize();
            if (offsetDir.length() < 0.1) offsetDir.set(0, 1, 1).normalize(); // Fallback for center star

            // Target camera position: offset from star
            targetCameraPos.copy(starPos).add(offsetDir.multiplyScalar(20)).add(new THREE.Vector3(0, 10, 0));

            isZooming = true;
            if (controls) controls.enabled = false; // Disable controls during animation

            // Update selection box
            selectionGroup.position.copy(starPos);
            selectionGroup.visible = true;

            // Draw connections
            drawConnectionsToNeighbors(selectedIndex);

            // Use Tween.js for smooth camera movement
            if (typeof TWEEN !== 'undefined') {
                // Move Camera Position
                new TWEEN.Tween(camera.position)
                    .to({ x: targetCameraPos.x, y: targetCameraPos.y, z: targetCameraPos.z }, 1500)
                    .easing(TWEEN.Easing.Cubic.InOut)
                    .onComplete(() => {
                        isZooming = false;
                        if (controls) {
                            controls.target.copy(starPos);
                            controls.enabled = true;
                            controls.update();
                        }
                    })
                    .start();

                // Move Controls Target (where the camera looks) smoothly too
                if (controls) {
                    new TWEEN.Tween(controls.target)
                        .to({ x: starPos.x, y: starPos.y, z: starPos.z }, 1000)
                        .easing(TWEEN.Easing.Cubic.InOut)
                        .start();
                }
            } else {
                // Fallback if tween.js is not loaded
                camera.position.copy(targetCameraPos);
                if (controls) {
                    controls.target.copy(starPos);
                    controls.enabled = true;
                }
                camera.lookAt(starPos);
                isZooming = false;
            }

        } else {
            // Clicked empty space: hide panel and reset
            hideInfoPanel();
            selectedIndex = null;
            selectionGroup.visible = false;
            connectionsGroup.clear(); // Clear lines

            if (controls) {
                controls.autoRotate = true;
                // Gently return target to center
                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(controls.target)
                        .to({ x: 0, y: 0, z: 0 }, 1000)
                        .easing(TWEEN.Easing.Quadratic.Out)
                        .start();
                }
            }
        }
    }

    function drawConnectionsToNeighbors(centerIndex, maxConnections = 4) {
        connectionsGroup.clear();

        const positions = points.geometry.attributes.position.array;
        const centerPos = new THREE.Vector3(
            positions[centerIndex * 3],
            positions[centerIndex * 3 + 1],
            positions[centerIndex * 3 + 2]
        );

        // Find nearest neighbors
        let distances = [];
        for (let i = 0; i < postsData.length; i++) {
            if (i === centerIndex) continue;
            const pos = new THREE.Vector3(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2]
            );
            distances.push({ index: i, pos: pos, dist: centerPos.distanceTo(pos) });
        }

        distances.sort((a, b) => a.dist - b.dist);
        const neighbors = distances.slice(0, maxConnections);

        // Draw lines
        const material = new THREE.LineBasicMaterial({
            color: 0x66CCFF,
            transparent: true,
            opacity: 0.8, // More visible
            linewidth: 3, // Thicker lines
            blending: THREE.AdditiveBlending
        });

        neighbors.forEach(n => {
            const geometry = new THREE.BufferGeometry().setFromPoints([centerPos, n.pos]);
            const line = new THREE.Line(geometry, material);
            connectionsGroup.add(line);
        });
    }

    function hideInfoPanel() {
        infoPanel.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if(infoPanel.style.transform === 'translateX(100%)') {
                infoPanel.style.display = 'none';
            }
        }, 400); // Wait for transition
    }

    window.addEventListener('mousemove', onMouseMove, false);
    window.addEventListener('click', onClick, false);

    // Handle Window Resize
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // 5. Animation
    let isAnimating = false;
    function animate(time) {
        if (!isAnimating) return;
        requestAnimationFrame(animate);

        if (typeof TWEEN !== 'undefined') {
            TWEEN.update(time);
        }

        if (selectionGroup && selectionGroup.visible && camera) {
            selectionGroup.lookAt(camera.position); // Selection box always faces camera
            selectionGroup.rotation.z += 0.01; // Spin slightly for tech effect
        }

        if (blackHoleMesh) {
            // Spin accretion disk slowly if not warping
            if (!isWarping) {
                blackHoleMesh.children[0].rotation.z -= 0.005;
            }
        }

        if (controls) {
            controls.update(); // Required for damping and autoRotate
        } else if (points && selectedIndex === null) {
            points.rotation.y += 0.001; // Fallback slow rotation
        }

        renderer.render(scene, camera);
    }

    // 6. Toggle Logic
    window.toggleStarMap = function() {
        if (container.style.display === 'none' || container.style.display === '') {
            container.style.display = 'block';
            isAnimating = true;
            isWarping = false;

            // Reset camera on open
            camera.position.set(0, 80, 150);
            if (controls) {
                controls.target.set(0,0,0);
                controls.enabled = true;
                controls.update();
            }
            hideInfoPanel();
            selectedIndex = null;

            // Restore stars if they were sucked into black hole
            if (points && initialPositions) {
                const positions = points.geometry.attributes.position.array;
                for (let i = 0; i < positions.length; i++) {
                    positions[i] = initialPositions[i];
                }
                points.geometry.attributes.position.needsUpdate = true;
            }

            animate(performance.now());

            // Add a close button if it doesn't exist
            let closeBtn = document.getElementById('starmap-close');
            if (!closeBtn) {
                closeBtn = document.createElement('div');
                closeBtn.id = 'starmap-close';
                closeBtn.innerHTML = '×';
                closeBtn.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10001; background: rgba(255,255,255,0.1); backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px); color: #fff; width: 40px; height: 40px; border-radius: 50%; text-align: center; line-height: 40px; font-size: 24px; cursor: pointer; transition: all 0.3s; border: 1px solid rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center;';
                closeBtn.onmouseover = function() { this.style.background = 'rgba(255,255,255,0.3)'; };
                closeBtn.onmouseout = function() { this.style.background = 'rgba(255,255,255,0.1)'; };
                closeBtn.onclick = window.toggleStarMap;
                document.body.appendChild(closeBtn);
            }
            closeBtn.style.display = 'flex';
        } else {
            container.style.display = 'none';
            isAnimating = false;
            hideInfoPanel();
            const closeBtn = document.getElementById('starmap-close');
            if (closeBtn) closeBtn.style.display = 'none';
        }
    };
});