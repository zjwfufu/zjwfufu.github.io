// Starmap: lazy-loaded Three.js galaxy visualization
// Three.js, OrbitControls, Tween.js are loaded on demand when starmap opens.

(function() {
    var starmapInitialized = false;
    var starmapReady = false;

    function loadScripts(urls, callback) {
        var i = 0;
        function next() {
            if (i >= urls.length) { callback(); return; }
            var s = document.createElement('script');
            s.src = urls[i];
            s.onload = function() { i++; next(); };
            s.onerror = function() { console.error('Failed to load: ' + urls[i]); };
            document.head.appendChild(s);
        }
        next();
    }

    window.toggleStarMap = function() {
        var container = document.getElementById('starmap-container');
        if (!container) return;

        if (container.style.display === 'none' || container.style.display === '') {
            container.style.display = 'block';

            // Show close button
            var closeBtn = document.getElementById('starmap-close');
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

            if (!starmapInitialized) {
                starmapInitialized = true;
                container.innerHTML = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:rgba(255,255,255,0.5);font-size:14px;letter-spacing:2px;">Loading starmap...</div>';

                loadScripts([
                    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
                    'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js',
                    'https://cdnjs.cloudflare.com/ajax/libs/tween.js/18.6.4/tween.umd.js'
                ], function() {
                    container.innerHTML = '';
                    initStarmap(container);
                    starmapReady = true;
                });
            } else if (starmapReady) {
                resumeStarmap();
            }
        } else {
            pauseStarmap(container);
        }
    };

    // =========================================================================
    // Original starmap core (visual logic unchanged)
    // =========================================================================

    var scene, camera, renderer, controls;
    var points, postsData = [];
    var dustPoints, dustMaterial;
    var blackHoleMesh, pointLight;
    var selectionGroup, connectionsGroup;
    var raycaster, mouse;
    var hoveredIndex = null, hoveredBlackHole = false, selectedIndex = null;
    var targetCameraPos;
    var isZooming = false, isWarping = false, isAnimating = false;
    var initialPositions = null;
    var containerEl;

    var infoPanel, infoTitle, infoMeta, infoExcerpt, infoLink;

    function initStarmap(container) {
        containerEl = container;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x020813);

        camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 60, 120);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = false;
            controls.maxDistance = 300;
            controls.minDistance = 5;
        }

        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();
        targetCameraPos = new THREE.Vector3();

        selectionGroup = new THREE.Group();
        scene.add(selectionGroup);
        createSelectionBox();

        connectionsGroup = new THREE.Group();
        scene.add(connectionsGroup);

        infoPanel = document.getElementById('starmap-info-panel');
        infoTitle = document.getElementById('starmap-info-title');
        infoMeta = document.getElementById('starmap-info-meta');
        infoExcerpt = document.getElementById('starmap-info-excerpt');
        infoLink = document.getElementById('starmap-info-link');

        fetch('/content.json')
            .then(function(response) { return response.json(); })
            .then(function(data) {
                var posts = data.posts || data;
                createGalaxy(posts);
            })
            .catch(function(err) { console.error("Error loading posts for star map:", err); });

        window.addEventListener('mousemove', onMouseMove, false);
        window.addEventListener('click', onClick, false);
        window.addEventListener('resize', onWindowResize, false);
    }

    function createCircleTexture(isCore) {
        var canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        var context = canvas.getContext('2d');
        var gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);

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

        var texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    function createSelectionBox() {
        var material = new THREE.LineBasicMaterial({ color: 0x66CCFF, linewidth: 2 });
        var size = 3;
        var length = 1;
        var pts = [
            [-size, size - length, 0], [-size, size, 0], [-size + length, size, 0],
            [size - length, size, 0], [size, size, 0], [size, size - length, 0],
            [size, -size + length, 0], [size, -size, 0], [size - length, -size, 0],
            [-size + length, -size, 0], [-size, -size, 0], [-size, -size + length, 0]
        ];
        for (var i = 0; i < 4; i++) {
            var geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(pts[i*3][0], pts[i*3][1], pts[i*3][2]),
                new THREE.Vector3(pts[i*3+1][0], pts[i*3+1][1], pts[i*3+1][2]),
                new THREE.Vector3(pts[i*3+2][0], pts[i*3+2][1], pts[i*3+2][2])
            ]);
            selectionGroup.add(new THREE.Line(geometry, material));
        }
        selectionGroup.visible = false;
    }

    function createGalaxy(posts) {
        postsData = posts;
        var particleCount = posts.length;

        var geometry = new THREE.BufferGeometry();
        var positions = new Float32Array(particleCount * 3);
        initialPositions = new Float32Array(particleCount * 3);
        var colors = new Float32Array(particleCount * 3);
        var sizes = new Float32Array(particleCount);

        var categoryMap = {};
        var branchCount = 0;

        posts.forEach(function(post) {
            var catName = 'Uncategorized';
            if (post.categories && post.categories.length > 0) {
                catName = post.categories[0].name;
            }
            if (categoryMap[catName] === undefined) {
                categoryMap[catName] = branchCount++;
            }
        });

        var totalBranches = Math.max(4, branchCount);
        var radius = 90;
        var spin = 1.5;
        var randomness = 15;
        var randomnessPower = 2.5;

        var baseBlue = new THREE.Color(0x33BBFF);

        var maxWords = 1;
        posts.forEach(function(post) {
            var wordCount = post.text ? post.text.length : (post.excerpt ? post.excerpt.length : 100);
            if (wordCount > maxWords) maxWords = wordCount;
        });

        for (var i = 0; i < particleCount; i++) {
            var post = posts[i];
            var catName = 'Uncategorized';
            if (post.categories && post.categories.length > 0) {
                catName = post.categories[0].name;
            }
            var branchIndex = categoryMap[catName];

            var radiusDistance = Math.pow(Math.random(), 1.2) * radius + 15;
            var branchAngle = ((branchIndex % totalBranches) / totalBranches) * Math.PI * 2;
            var spinAngle = radiusDistance * spin / radius;

            var randomX = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;
            var randomY = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;
            var randomZ = Math.pow(Math.random(), randomnessPower) * (Math.random() < 0.5 ? 1 : -1) * randomness * (radius - radiusDistance) / radius;

            positions[i * 3]     = Math.cos(branchAngle + spinAngle) * radiusDistance + randomX;
            positions[i * 3 + 1] = randomY * 8.0;
            positions[i * 3 + 2] = Math.sin(branchAngle + spinAngle) * radiusDistance + randomZ;

            initialPositions[i * 3]     = positions[i * 3];
            initialPositions[i * 3 + 1] = positions[i * 3 + 1];
            initialPositions[i * 3 + 2] = positions[i * 3 + 2];

            var baseColor = baseBlue.clone();
            var hsl = {};
            baseColor.getHSL(hsl);
            var lightness = Math.max(0.6, Math.min(1.0, 1.0 - (radiusDistance / radius) * 0.3 + (Math.random() - 0.5) * 0.1));
            baseColor.setHSL(
                hsl.h + (Math.random() - 0.5) * 0.05,
                Math.max(0.8, Math.min(1.0, hsl.s + (Math.random() - 0.5) * 0.1)),
                lightness
            );

            colors[i * 3]     = baseColor.r;
            colors[i * 3 + 1] = baseColor.g;
            colors[i * 3 + 2] = baseColor.b;

            var wordCount = post.text ? post.text.length : (post.excerpt ? post.excerpt.length : 100);
            sizes[i] = 6 + (wordCount / maxWords) * 12;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        var vertexShader = [
            'attribute float size;',
            'varying vec3 vColor;',
            'void main() {',
            '    vColor = color;',
            '    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
            '    gl_PointSize = size * (300.0 / -mvPosition.z);',
            '    gl_Position = projectionMatrix * mvPosition;',
            '}'
        ].join('\n');

        var fragmentShader = [
            'uniform sampler2D pointTexture;',
            'varying vec3 vColor;',
            'void main() {',
            '    gl_FragColor = vec4(vColor, 1.0) * texture2D(pointTexture, gl_PointCoord);',
            '}'
        ].join('\n');

        var material = new THREE.ShaderMaterial({
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

        createCosmicDust();
        createBlackHole();

        isAnimating = true;
        animate(performance.now());
    }

    function createBlackHole() {
        var accretionGeometry = new THREE.RingGeometry(4, 12, 64);
        var accretionMaterial = new THREE.MeshBasicMaterial({
            color: 0x33BBFF,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            map: createCircleTexture(true)
        });
        var accretionDisk = new THREE.Mesh(accretionGeometry, accretionMaterial);
        accretionDisk.rotation.x = Math.PI / 2;

        var bhGeometry = new THREE.SphereGeometry(3.8, 32, 32);
        var bhMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        var bhSphere = new THREE.Mesh(bhGeometry, bhMaterial);

        pointLight = new THREE.PointLight(0xddf3ff, 2.5, 100);
        pointLight.position.set(0, 0, 0);

        var glowParticlesCount = 600;
        var glowGeometry = new THREE.BufferGeometry();
        var glowPositions = new Float32Array(glowParticlesCount * 3);
        var glowColors = new Float32Array(glowParticlesCount * 3);

        var innerColor = new THREE.Color(0xddf3ff);
        var outerColor = new THREE.Color(0x1166ff);

        for (var i = 0; i < glowParticlesCount; i++) {
            var r = Math.pow(Math.random(), 1.8) * 80 + 3;
            var theta = Math.random() * Math.PI * 2;
            glowPositions[i * 3] = r * Math.cos(theta);
            glowPositions[i * 3 + 1] = (Math.random() - 0.5) * 1.5;
            glowPositions[i * 3 + 2] = r * Math.sin(theta);
            var interpColor = innerColor.clone().lerp(outerColor, r / 80);
            glowColors[i * 3] = interpColor.r;
            glowColors[i * 3 + 1] = interpColor.g;
            glowColors[i * 3 + 2] = interpColor.b;
        }

        glowGeometry.setAttribute('position', new THREE.BufferAttribute(glowPositions, 3));
        glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowColors, 3));

        var glowMaterial = new THREE.PointsMaterial({
            size: 15.0,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.35,
            blending: THREE.AdditiveBlending,
            map: createCircleTexture(true),
            depthWrite: false,
            vertexColors: true
        });

        var bhGlowPoints = new THREE.Points(glowGeometry, glowMaterial);

        blackHoleMesh = new THREE.Group();
        blackHoleMesh.add(accretionDisk);  // 0
        blackHoleMesh.add(bhGlowPoints);   // 1
        blackHoleMesh.add(bhSphere);       // 2
        blackHoleMesh.add(pointLight);     // 3

        var hitGeometry = new THREE.SphereGeometry(8, 16, 16);
        var hitMaterial = new THREE.MeshBasicMaterial({ visible: false });
        var hitBox = new THREE.Mesh(hitGeometry, hitMaterial);
        hitBox.userData = { isBlackHole: true };
        blackHoleMesh.add(hitBox);          // 4

        scene.add(blackHoleMesh);
    }

    function createCosmicDust() {
        var dustCount = 8000;
        var dustGeo = new THREE.BufferGeometry();
        var dustPos = new Float32Array(dustCount * 3);
        var dustCol = new Float32Array(dustCount * 3);
        var dustSizes = new Float32Array(dustCount);

        var baseBlue = new THREE.Color(0x2288CC);

        for (var i = 0; i < dustCount; i++) {
            var r = Math.random() * 360 + 40;
            var theta = Math.random() * Math.PI * 2;
            var phi = Math.acos((Math.random() * 2) - 1);

            dustPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
            dustPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
            dustPos[i*3+2] = r * Math.cos(phi);

            var color = baseBlue.clone();
            var hsl = {};
            color.getHSL(hsl);
            var lightness = Math.random() * 0.4 + 0.1;
            color.setHSL(hsl.h + (Math.random() - 0.5) * 0.05, hsl.s, lightness);

            dustCol[i*3] = color.r;
            dustCol[i*3+1] = color.g;
            dustCol[i*3+2] = color.b;

            dustSizes[i] = Math.random() * Math.PI * 2;
        }

        dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
        dustGeo.setAttribute('color', new THREE.BufferAttribute(dustCol, 3));
        dustGeo.setAttribute('twinkleOffset', new THREE.BufferAttribute(dustSizes, 1));

        var dustVertexShader = [
            'attribute float twinkleOffset;',
            'varying vec3 vColor;',
            'varying float vTwinkle;',
            'uniform float uTime;',
            'void main() {',
            '    vColor = color;',
            '    vTwinkle = 0.3 + 0.7 * sin(uTime * 2.0 + twinkleOffset);',
            '    vec3 pos = position;',
            '    pos.x += sin(uTime * 0.3 + twinkleOffset) * 12.0;',
            '    pos.y += cos(uTime * 0.2 + twinkleOffset) * 12.0;',
            '    pos.z += sin(uTime * 0.1 + twinkleOffset) * 12.0;',
            '    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);',
            '    float baseSize = 0.8 + fract(twinkleOffset * 3.17) * 2.5;',
            '    gl_PointSize = baseSize * (300.0 / -mvPosition.z);',
            '    gl_Position = projectionMatrix * mvPosition;',
            '}'
        ].join('\n');

        var dustFragmentShader = [
            'uniform sampler2D pointTexture;',
            'varying vec3 vColor;',
            'varying float vTwinkle;',
            'void main() {',
            '    gl_FragColor = vec4(vColor * vTwinkle, 0.6) * texture2D(pointTexture, gl_PointCoord);',
            '}'
        ].join('\n');

        dustMaterial = new THREE.ShaderMaterial({
            uniforms: {
                pointTexture: { value: createCircleTexture(true) },
                uTime: { value: 0.0 }
            },
            vertexShader: dustVertexShader,
            fragmentShader: dustFragmentShader,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true,
            vertexColors: true
        });

        dustPoints = new THREE.Points(dustGeo, dustMaterial);
        scene.add(dustPoints);
    }

    // =========================================================================
    // Interactivity
    // =========================================================================

    function onMouseMove(event) {
        if (!containerEl || containerEl.style.display === 'none') return;
        if (isZooming || isWarping) return;

        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        if (blackHoleMesh) {
            raycaster.setFromCamera(mouse, camera);
            var hitBox = blackHoleMesh.children[4];
            var bhIntersects = raycaster.intersectObject(hitBox);
            if (bhIntersects.length > 0) {
                hoveredBlackHole = true;
                containerEl.style.cursor = 'pointer';
                if (controls) controls.autoRotate = false;
                hoveredIndex = null;
                return;
            }
        }
        hoveredBlackHole = false;

        if (points) {
            raycaster.params.Points.threshold = 2.0;
            raycaster.setFromCamera(mouse, camera);
            var intersects = raycaster.intersectObject(points);

            if (intersects.length > 0) {
                var intersect = intersects[0];
                if (hoveredIndex !== intersect.index) {
                    hoveredIndex = intersect.index;
                    containerEl.style.cursor = 'pointer';
                    if (controls) controls.autoRotate = false;
                }
            } else {
                hoveredIndex = null;
                if (!hoveredBlackHole) {
                    containerEl.style.cursor = 'default';
                    if (controls && selectedIndex === null) controls.autoRotate = true;
                }
            }
        }
    }

    function triggerWarp() {
        isWarping = true;
        if (controls) controls.enabled = false;
        hideInfoPanel();
        selectionGroup.visible = false;
        connectionsGroup.clear();

        if (typeof TWEEN !== 'undefined') {
            new TWEEN.Tween(camera.position)
                .to({ x: 0, y: 5, z: 0 }, 2500)
                .easing(TWEEN.Easing.Cubic.In)
                .start();
            if (controls) {
                new TWEEN.Tween(controls.target).to({ x: 0, y: 0, z: 0 }, 1000).start();
            }
        }

        var positions = points.geometry.attributes.position.array;
        var particleCount = positions.length / 3;

        for (var i = 0; i < particleCount; i++) {
            (function(idx) {
                var startPos = { x: positions[idx*3], y: positions[idx*3+1], z: positions[idx*3+2] };
                var delay = Math.random() * 1000;
                var duration = 1000 + Math.random() * 1000;
                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(startPos)
                        .to({ x: 0, y: 0, z: 0 }, duration)
                        .delay(delay)
                        .easing(TWEEN.Easing.Quadratic.In)
                        .onUpdate(function() {
                            positions[idx*3] = startPos.x;
                            positions[idx*3+1] = startPos.y;
                            positions[idx*3+2] = startPos.z;
                            points.geometry.attributes.position.needsUpdate = true;
                        })
                        .start();
                }
            })(i);
        }

        if (typeof TWEEN !== 'undefined') {
            var rotationTween = { speed: 0.05 };
            new TWEEN.Tween(rotationTween)
                .to({ speed: 0.5 }, 2000)
                .onUpdate(function() {
                    if (blackHoleMesh) blackHoleMesh.children[0].rotation.z += rotationTween.speed;
                })
                .start();
        }

        setTimeout(function() { window.toggleStarMap(); }, 2600);
    }

    function onClick(event) {
        if (!containerEl || containerEl.style.display === 'none') return;
        if (isWarping) return;
        if (event.target && event.target.id === 'starmap-close') return;
        if (event.target && event.target.closest && event.target.closest('#starmap-info-panel')) return;

        if (hoveredBlackHole) {
            infoTitle.innerHTML = '？？？';
            infoMeta.innerHTML = '';
            infoExcerpt.innerHTML = '？？？';
            infoLink.innerHTML = '确认折跃';
            infoLink.href = 'javascript:void(0)';
            infoLink.onclick = function(e) { e.preventDefault(); triggerWarp(); };
            infoPanel.style.display = 'block';
            setTimeout(function() { infoPanel.style.transform = 'translateX(0)'; }, 10);
            selectionGroup.position.set(0, 0, 0);
            selectionGroup.visible = true;
            selectedIndex = -1;
            return;
        }

        if (hoveredIndex !== null && postsData[hoveredIndex]) {
            selectedIndex = hoveredIndex;
            var post = postsData[hoveredIndex];

            infoTitle.innerHTML = post.title || 'Untitled';

            var metaHtml = '';
            if (post.date) {
                var date = new Date(post.date);
                metaHtml += '<span>\u{1F4C5} ' + date.getFullYear() + '-' + String(date.getMonth()+1).padStart(2,'0') + '-' + String(date.getDate()).padStart(2,'0') + '</span>';
            }
            if (post.categories && post.categories.length > 0) {
                metaHtml += '<span>\u{1F4C1} ' + post.categories[0].name + '</span>';
            }
            infoMeta.innerHTML = metaHtml;

            var excerptStr = post.excerpt || post.text || '浩瀚宇宙中，这颗星星上还没有留下摘录...';
            excerptStr = excerptStr.replace(/<[^>]+>/g, '').substring(0, 150) + '...';
            infoExcerpt.innerHTML = excerptStr;

            var rootPath = typeof yiliaConfig !== 'undefined' && yiliaConfig.root ? yiliaConfig.root : '/';
            var cleanPath = post.path.startsWith('/') ? post.path.substring(1) : post.path;
            var cleanRoot = rootPath.endsWith('/') ? rootPath : rootPath + '/';
            infoLink.innerHTML = '折跃 &rarr;';
            infoLink.href = cleanRoot + cleanPath;
            infoLink.onclick = null;

            infoPanel.style.display = 'block';
            setTimeout(function() { infoPanel.style.transform = 'translateX(0)'; }, 10);

            var pos = points.geometry.attributes.position.array;
            var targetX = pos[selectedIndex * 3];
            var targetY = pos[selectedIndex * 3 + 1];
            var targetZ = pos[selectedIndex * 3 + 2];
            var starPos = new THREE.Vector3(targetX, targetY, targetZ);

            var offsetDir = starPos.clone().normalize();
            if (offsetDir.length() < 0.1) offsetDir.set(0, 1, 1).normalize();
            targetCameraPos.copy(starPos).add(offsetDir.multiplyScalar(20)).add(new THREE.Vector3(0, 10, 0));

            isZooming = true;
            if (controls) controls.enabled = false;

            selectionGroup.position.copy(starPos);
            selectionGroup.visible = true;
            drawConnectionsToNeighbors(selectedIndex);

            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(camera.position)
                    .to({ x: targetCameraPos.x, y: targetCameraPos.y, z: targetCameraPos.z }, 1500)
                    .easing(TWEEN.Easing.Cubic.InOut)
                    .onComplete(function() {
                        isZooming = false;
                        if (controls) {
                            controls.target.copy(starPos);
                            controls.enabled = true;
                            controls.update();
                        }
                    })
                    .start();
                if (controls) {
                    new TWEEN.Tween(controls.target)
                        .to({ x: starPos.x, y: starPos.y, z: starPos.z }, 1000)
                        .easing(TWEEN.Easing.Cubic.InOut)
                        .start();
                }
            } else {
                camera.position.copy(targetCameraPos);
                if (controls) { controls.target.copy(starPos); controls.enabled = true; }
                camera.lookAt(starPos);
                isZooming = false;
            }

        } else {
            hideInfoPanel();
            selectedIndex = null;
            selectionGroup.visible = false;
            connectionsGroup.clear();
            if (controls) {
                controls.autoRotate = true;
                controls.enabled = true;
                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(controls.target)
                        .to({ x: 0, y: 0, z: 0 }, 1000)
                        .easing(TWEEN.Easing.Quadratic.Out)
                        .start();
                }
            }
        }
    }

    function drawConnectionsToNeighbors(centerIndex, maxConnections) {
        maxConnections = maxConnections || 4;
        connectionsGroup.clear();

        var positions = points.geometry.attributes.position.array;
        var centerPos = new THREE.Vector3(positions[centerIndex*3], positions[centerIndex*3+1], positions[centerIndex*3+2]);

        var distances = [];
        for (var i = 0; i < postsData.length; i++) {
            if (i === centerIndex) continue;
            var pos = new THREE.Vector3(positions[i*3], positions[i*3+1], positions[i*3+2]);
            distances.push({ index: i, pos: pos, dist: centerPos.distanceTo(pos) });
        }
        distances.sort(function(a, b) { return a.dist - b.dist; });
        var neighbors = distances.slice(0, maxConnections);

        var material = new THREE.LineBasicMaterial({
            color: 0x66CCFF,
            transparent: true,
            opacity: 0.8,
            linewidth: 3,
            blending: THREE.AdditiveBlending
        });

        neighbors.forEach(function(n) {
            var geometry = new THREE.BufferGeometry().setFromPoints([centerPos, n.pos]);
            connectionsGroup.add(new THREE.Line(geometry, material));
        });
    }

    function hideInfoPanel() {
        if (!infoPanel) return;
        infoPanel.style.transform = 'translateX(100%)';
        setTimeout(function() {
            if (infoPanel.style.transform === 'translateX(100%)') infoPanel.style.display = 'none';
        }, 400);
    }

    // =========================================================================
    // Animation
    // =========================================================================

    function animate(time) {
        if (!isAnimating) return;
        requestAnimationFrame(animate);

        if (typeof TWEEN !== 'undefined') TWEEN.update(time);

        if (selectionGroup && selectionGroup.visible && camera) {
            selectionGroup.lookAt(camera.position);
            selectionGroup.rotation.z += 0.01;
        }

        if (dustMaterial) {
            dustMaterial.uniforms.uTime.value = time * 0.001;
        }

        if (blackHoleMesh) {
            if (!isWarping) {
                blackHoleMesh.children[0].rotation.z -= 0.005;
                blackHoleMesh.children[1].rotation.y += 0.002;
            }
        }

        if (controls) {
            controls.update();
        } else if (points && selectedIndex === null) {
            points.rotation.y += 0.001;
        }

        if (dustPoints && selectedIndex === null) {
            dustPoints.rotation.y += 0.0003;
            dustPoints.rotation.x += 0.0001;
            dustPoints.rotation.z += 0.0002;
        }

        renderer.render(scene, camera);
    }

    function onWindowResize() {
        if (!camera || !renderer) return;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // =========================================================================
    // Toggle helpers
    // =========================================================================

    function resumeStarmap() {
        isAnimating = true;
        isWarping = false;

        camera.position.set(0, 80, 150);
        if (controls) {
            controls.target.set(0, 0, 0);
            controls.enabled = true;
            controls.update();
        }
        hideInfoPanel();
        selectedIndex = null;

        if (points && initialPositions) {
            var positions = points.geometry.attributes.position.array;
            for (var i = 0; i < positions.length; i++) {
                positions[i] = initialPositions[i];
            }
            points.geometry.attributes.position.needsUpdate = true;
        }

        animate(performance.now());
    }

    function pauseStarmap(container) {
        container.style.display = 'none';
        isAnimating = false;
        hideInfoPanel();
        var closeBtn = document.getElementById('starmap-close');
        if (closeBtn) closeBtn.style.display = 'none';
    }

})();
