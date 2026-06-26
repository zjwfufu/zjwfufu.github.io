// Starmap: lazy-loaded Three.js galaxy visualization
// Three.js, OrbitControls, Tween.js are loaded on demand when starmap opens.

(function() {
    var starmapInitialized = false;
    var starmapReady = false;

    // Lazy-load scripts sequentially, then call callback
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
                closeBtn.style.cssText = 'position:fixed;top:20px;right:20px;z-index:10001;background:rgba(255,255,255,0.1);backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px);color:#fff;width:40px;height:40px;border-radius:50%;text-align:center;line-height:40px;font-size:24px;cursor:pointer;transition:all 0.3s;border:1px solid rgba(255,255,255,0.2);display:flex;align-items:center;justify-content:center;';
                closeBtn.onmouseover = function() { this.style.background = 'rgba(255,255,255,0.3)'; };
                closeBtn.onmouseout = function() { this.style.background = 'rgba(255,255,255,0.1)'; };
                closeBtn.onclick = window.toggleStarMap;
                document.body.appendChild(closeBtn);
            }
            closeBtn.style.display = 'flex';

            if (!starmapInitialized) {
                starmapInitialized = true;
                // Show loading indicator
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
    // Starmap core
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
    var shootingStars = [];
    var containerEl;

    // UI elements
    var infoPanel, infoTitle, infoMeta, infoExcerpt, infoLink;

    // Stellar spectral color palette — real star colors mapped to categories
    var STAR_PALETTE = [
        new THREE.Color(0x9db4ff), // O-type: blue-white
        new THREE.Color(0xaabfff), // B-type: blue-white
        new THREE.Color(0xcad8ff), // A-type: white
        new THREE.Color(0xfff4e8), // F-type: yellow-white
        new THREE.Color(0xffd2a1), // G-type: warm yellow (sun-like)
        new THREE.Color(0xffb46b), // K-type: orange
        new THREE.Color(0xff6060), // M-type: red
        new THREE.Color(0xb8c7ff), // extra: soft blue
        new THREE.Color(0xe0c8ff), // extra: lavender
        new THREE.Color(0x80e0ff), // extra: cyan
        new THREE.Color(0x90ffa0), // extra: green nebula
        new THREE.Color(0xffa0d0), // extra: pink
    ];

    function initStarmap(container) {
        containerEl = container;
        scene = new THREE.Scene();
        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();
        targetCameraPos = new THREE.Vector3();

        // Nebula background texture
        scene.background = createNebulaBackground();

        camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 80, 150);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.3;
            controls.maxDistance = 300;
            controls.minDistance = 5;
        }

        // Selection box
        selectionGroup = new THREE.Group();
        scene.add(selectionGroup);
        createSelectionBox();

        // Connections
        connectionsGroup = new THREE.Group();
        scene.add(connectionsGroup);

        // UI references
        infoPanel = document.getElementById('starmap-info-panel');
        infoTitle = document.getElementById('starmap-info-title');
        infoMeta = document.getElementById('starmap-info-meta');
        infoExcerpt = document.getElementById('starmap-info-excerpt');
        infoLink = document.getElementById('starmap-info-link');

        // Fetch data
        fetch('/content.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var posts = data.posts || data;
                createGalaxy(posts);
            })
            .catch(function(err) { console.error("Error loading posts for star map:", err); });

        // Event listeners
        window.addEventListener('mousemove', onMouseMove, false);
        window.addEventListener('click', onClick, false);
        window.addEventListener('resize', onWindowResize, false);
    }

    // =========================================================================
    // Nebula Background
    // =========================================================================
    function createNebulaBackground() {
        var canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        var ctx = canvas.getContext('2d');

        // Deep space base
        ctx.fillStyle = '#050510';
        ctx.fillRect(0, 0, 1024, 1024);

        // Nebula clouds — large soft radial gradients
        var nebulae = [
            { x: 200, y: 400, r: 500, color: [30, 10, 60, 0.15] },   // purple nebula
            { x: 700, y: 300, r: 450, color: [10, 30, 80, 0.12] },   // blue nebula
            { x: 500, y: 700, r: 400, color: [10, 50, 60, 0.10] },   // teal nebula
            { x: 150, y: 150, r: 350, color: [50, 10, 40, 0.08] },   // magenta hint
            { x: 800, y: 800, r: 300, color: [20, 20, 60, 0.10] },   // deep blue
            { x: 512, y: 512, r: 600, color: [15, 15, 40, 0.12] },   // center glow
        ];

        nebulae.forEach(function(n) {
            var g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r);
            g.addColorStop(0, 'rgba(' + n.color[0] + ',' + n.color[1] + ',' + n.color[2] + ',' + n.color[3] + ')');
            g.addColorStop(0.5, 'rgba(' + n.color[0] + ',' + n.color[1] + ',' + n.color[2] + ',' + (n.color[3] * 0.4) + ')');
            g.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g;
            ctx.fillRect(0, 0, 1024, 1024);
        });

        // Galactic plane — horizontal bright band
        var bandGrad = ctx.createLinearGradient(0, 400, 0, 624);
        bandGrad.addColorStop(0, 'rgba(0,0,0,0)');
        bandGrad.addColorStop(0.3, 'rgba(20,15,50,0.08)');
        bandGrad.addColorStop(0.5, 'rgba(40,30,80,0.12)');
        bandGrad.addColorStop(0.7, 'rgba(20,15,50,0.08)');
        bandGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = bandGrad;
        ctx.fillRect(0, 0, 1024, 1024);

        // Scatter tiny static background stars on the texture
        for (var i = 0; i < 800; i++) {
            var sx = Math.random() * 1024;
            var sy = Math.random() * 1024;
            var brightness = Math.random() * 0.6 + 0.2;
            var size = Math.random() * 1.5 + 0.5;
            ctx.fillStyle = 'rgba(255,255,255,' + brightness + ')';
            ctx.beginPath();
            ctx.arc(sx, sy, size, 0, Math.PI * 2);
            ctx.fill();
        }

        var texture = new THREE.CanvasTexture(canvas);
        texture.mapping = THREE.EquirectangularReflectionMapping;
        return texture;
    }

    // =========================================================================
    // Star Texture
    // =========================================================================
    function createCircleTexture(isCore) {
        var canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        var ctx = canvas.getContext('2d');

        var g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
        if (isCore) {
            g.addColorStop(0, 'rgba(255,255,255,1)');
            g.addColorStop(0.08, 'rgba(255,255,255,0.9)');
            g.addColorStop(0.25, 'rgba(255,255,255,0.3)');
            g.addColorStop(0.5, 'rgba(255,255,255,0.05)');
            g.addColorStop(1, 'rgba(0,0,0,0)');
        } else {
            // Star glow: tight bright core + wide soft halo
            g.addColorStop(0, 'rgba(255,255,255,1)');
            g.addColorStop(0.05, 'rgba(255,255,255,1)');
            g.addColorStop(0.15, 'rgba(255,255,255,0.7)');
            g.addColorStop(0.35, 'rgba(255,255,255,0.2)');
            g.addColorStop(0.6, 'rgba(255,255,255,0.05)');
            g.addColorStop(1, 'rgba(0,0,0,0)');
        }

        ctx.fillStyle = g;
        ctx.fillRect(0, 0, 128, 128);

        var texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    // =========================================================================
    // Selection Box
    // =========================================================================
    function createSelectionBox() {
        var material = new THREE.LineBasicMaterial({ color: 0x66CCFF, linewidth: 2 });
        var size = 3, length = 1;
        var pts = [
            [-size, size - length, 0], [-size, size, 0], [-size + length, size, 0],
            [size - length, size, 0], [size, size, 0], [size, size - length, 0],
            [size, -size + length, 0], [size, -size, 0], [size - length, -size, 0],
            [-size + length, -size, 0], [-size, -size, 0], [-size, -size + length, 0]
        ];
        for (var i = 0; i < 4; i++) {
            var geo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(pts[i*3][0], pts[i*3][1], pts[i*3][2]),
                new THREE.Vector3(pts[i*3+1][0], pts[i*3+1][1], pts[i*3+1][2]),
                new THREE.Vector3(pts[i*3+2][0], pts[i*3+2][1], pts[i*3+2][2])
            ]);
            selectionGroup.add(new THREE.Line(geo, material));
        }
        selectionGroup.visible = false;
    }

    // =========================================================================
    // Galaxy
    // =========================================================================
    function createGalaxy(posts) {
        postsData = posts;
        var count = posts.length;

        var geometry = new THREE.BufferGeometry();
        var positions = new Float32Array(count * 3);
        initialPositions = new Float32Array(count * 3);
        var colors = new Float32Array(count * 3);
        var sizes = new Float32Array(count);
        var twinkleOffsets = new Float32Array(count);

        // Map categories to branches
        var categoryMap = {};
        var branchCount = 0;
        posts.forEach(function(post) {
            var cat = (post.categories && post.categories.length > 0) ? post.categories[0].name : 'Uncategorized';
            if (categoryMap[cat] === undefined) categoryMap[cat] = branchCount++;
        });

        var totalBranches = Math.max(4, branchCount);
        var radius = 90;
        var spin = 1.5;
        var randomness = 15;
        var randomnessPower = 2.5;

        // Find max word count for sizing
        var maxWords = 1;
        posts.forEach(function(post) {
            var wc = post.excerpt ? post.excerpt.length : 100;
            if (wc > maxWords) maxWords = wc;
        });

        for (var i = 0; i < count; i++) {
            var post = posts[i];
            var cat = (post.categories && post.categories.length > 0) ? post.categories[0].name : 'Uncategorized';
            var branchIndex = categoryMap[cat];

            var radiusDistance = Math.pow(Math.random(), 1.2) * radius + 15;
            var branchAngle = ((branchIndex % totalBranches) / totalBranches) * Math.PI * 2;
            var spinAngle = radiusDistance * spin / radius;

            var rp = randomnessPower;
            var rf = randomness * (radius - radiusDistance) / radius;
            var randomX = Math.pow(Math.random(), rp) * (Math.random() < 0.5 ? 1 : -1) * rf;
            var randomY = Math.pow(Math.random(), rp) * (Math.random() < 0.5 ? 1 : -1) * rf;
            var randomZ = Math.pow(Math.random(), rp) * (Math.random() < 0.5 ? 1 : -1) * rf;

            positions[i*3]     = Math.cos(branchAngle + spinAngle) * radiusDistance + randomX;
            positions[i*3 + 1] = randomY * 8.0;
            positions[i*3 + 2] = Math.sin(branchAngle + spinAngle) * radiusDistance + randomZ;

            initialPositions[i*3]     = positions[i*3];
            initialPositions[i*3 + 1] = positions[i*3 + 1];
            initialPositions[i*3 + 2] = positions[i*3 + 2];

            // Color from stellar palette based on category
            var baseColor = STAR_PALETTE[branchIndex % STAR_PALETTE.length].clone();
            var hsl = {};
            baseColor.getHSL(hsl);

            // Brighter near center
            var lightness = Math.max(0.6, Math.min(1.0, 1.0 - (radiusDistance / radius) * 0.3 + (Math.random() - 0.5) * 0.1));
            baseColor.setHSL(
                hsl.h + (Math.random() - 0.5) * 0.03,
                Math.max(0.5, Math.min(1.0, hsl.s + (Math.random() - 0.5) * 0.15)),
                lightness
            );

            colors[i*3]     = baseColor.r;
            colors[i*3 + 1] = baseColor.g;
            colors[i*3 + 2] = baseColor.b;

            // Size based on excerpt length
            var wc = post.excerpt ? post.excerpt.length : 100;
            sizes[i] = 6 + (wc / maxWords) * 14;

            // Random twinkle phase
            twinkleOffsets[i] = Math.random() * Math.PI * 2;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('twinkleOffset', new THREE.BufferAttribute(twinkleOffsets, 1));

        // Custom shader with twinkling
        var vertexShader = [
            'attribute float size;',
            'attribute float twinkleOffset;',
            'varying vec3 vColor;',
            'varying float vTwinkle;',
            'uniform float uTime;',
            'void main() {',
            '    vColor = color;',
            '    vTwinkle = 0.7 + 0.3 * sin(uTime * 1.5 + twinkleOffset);',
            '    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
            '    gl_PointSize = size * (300.0 / -mvPosition.z);',
            '    gl_Position = projectionMatrix * mvPosition;',
            '}'
        ].join('\n');

        var fragmentShader = [
            'uniform sampler2D pointTexture;',
            'varying vec3 vColor;',
            'varying float vTwinkle;',
            'void main() {',
            '    gl_FragColor = vec4(vColor * vTwinkle, 1.0) * texture2D(pointTexture, gl_PointCoord);',
            '}'
        ].join('\n');

        var material = new THREE.ShaderMaterial({
            uniforms: {
                pointTexture: { value: createCircleTexture(false) },
                uTime: { value: 0.0 }
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

    // =========================================================================
    // Black Hole
    // =========================================================================
    function createBlackHole() {
        var accretionGeo = new THREE.RingGeometry(4, 12, 64);
        var accretionMat = new THREE.MeshBasicMaterial({
            color: 0x33BBFF,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            map: createCircleTexture(true)
        });
        var accretionDisk = new THREE.Mesh(accretionGeo, accretionMat);
        accretionDisk.rotation.x = Math.PI / 2;

        var bhGeo = new THREE.SphereGeometry(3.8, 32, 32);
        var bhMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
        var bhSphere = new THREE.Mesh(bhGeo, bhMat);

        pointLight = new THREE.PointLight(0xddf3ff, 2.5, 100);
        pointLight.position.set(0, 0, 0);

        // Glow cloud
        var glowCount = 600;
        var glowGeo = new THREE.BufferGeometry();
        var glowPos = new Float32Array(glowCount * 3);
        var glowCol = new Float32Array(glowCount * 3);
        var innerColor = new THREE.Color(0xddf3ff);
        var outerColor = new THREE.Color(0x1166ff);

        for (var i = 0; i < glowCount; i++) {
            var r = Math.pow(Math.random(), 1.8) * 80 + 3;
            var theta = Math.random() * Math.PI * 2;
            glowPos[i*3] = r * Math.cos(theta);
            glowPos[i*3+1] = (Math.random() - 0.5) * 1.5;
            glowPos[i*3+2] = r * Math.sin(theta);
            var interpColor = innerColor.clone().lerp(outerColor, r / 80);
            glowCol[i*3] = interpColor.r;
            glowCol[i*3+1] = interpColor.g;
            glowCol[i*3+2] = interpColor.b;
        }

        glowGeo.setAttribute('position', new THREE.BufferAttribute(glowPos, 3));
        glowGeo.setAttribute('color', new THREE.BufferAttribute(glowCol, 3));

        var glowMat = new THREE.PointsMaterial({
            size: 15.0,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.35,
            blending: THREE.AdditiveBlending,
            map: createCircleTexture(true),
            depthWrite: false,
            vertexColors: true
        });

        var bhGlowPoints = new THREE.Points(glowGeo, glowMat);

        blackHoleMesh = new THREE.Group();
        blackHoleMesh.add(accretionDisk);  // 0
        blackHoleMesh.add(bhGlowPoints);   // 1
        blackHoleMesh.add(bhSphere);       // 2
        blackHoleMesh.add(pointLight);     // 3

        // Invisible hit box
        var hitGeo = new THREE.SphereGeometry(8, 16, 16);
        var hitMat = new THREE.MeshBasicMaterial({ visible: false });
        var hitBox = new THREE.Mesh(hitGeo, hitMat);
        hitBox.userData = { isBlackHole: true };
        blackHoleMesh.add(hitBox);          // 4

        scene.add(blackHoleMesh);
    }

    // =========================================================================
    // Cosmic Dust
    // =========================================================================
    function createCosmicDust() {
        var dustCount = 8000;
        var dustGeo = new THREE.BufferGeometry();
        var dustPos = new Float32Array(dustCount * 3);
        var dustCol = new Float32Array(dustCount * 3);
        var dustTwinkle = new Float32Array(dustCount);

        // Mix of blue, purple, warm dust colors
        var dustColors = [
            new THREE.Color(0x2288CC),
            new THREE.Color(0x3355AA),
            new THREE.Color(0x6644AA),
            new THREE.Color(0x2266AA),
            new THREE.Color(0x557799),
        ];

        for (var i = 0; i < dustCount; i++) {
            var r = Math.random() * 360 + 40;
            var theta = Math.random() * Math.PI * 2;
            var phi = Math.acos((Math.random() * 2) - 1);

            dustPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
            dustPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
            dustPos[i*3+2] = r * Math.cos(phi);

            var color = dustColors[Math.floor(Math.random() * dustColors.length)].clone();
            var hsl = {};
            color.getHSL(hsl);
            color.setHSL(hsl.h + (Math.random() - 0.5) * 0.08, hsl.s, Math.random() * 0.4 + 0.1);

            dustCol[i*3] = color.r;
            dustCol[i*3+1] = color.g;
            dustCol[i*3+2] = color.b;

            dustTwinkle[i] = Math.random() * Math.PI * 2;
        }

        dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
        dustGeo.setAttribute('color', new THREE.BufferAttribute(dustCol, 3));
        dustGeo.setAttribute('twinkleOffset', new THREE.BufferAttribute(dustTwinkle, 1));

        var dustVS = [
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
            '    gl_PointSize = (0.5 + twinkleOffset * 0.4) * (300.0 / -mvPosition.z);',
            '    gl_Position = projectionMatrix * mvPosition;',
            '}'
        ].join('\n');

        var dustFS = [
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
            vertexShader: dustVS,
            fragmentShader: dustFS,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true,
            vertexColors: true
        });

        dustPoints = new THREE.Points(dustGeo, dustMaterial);
        scene.add(dustPoints);
    }

    // =========================================================================
    // Shooting Stars
    // =========================================================================
    function spawnShootingStar() {
        // Random start position far from center
        var startAngle = Math.random() * Math.PI * 2;
        var startR = 100 + Math.random() * 150;
        var startY = (Math.random() - 0.5) * 80;
        var start = new THREE.Vector3(
            Math.cos(startAngle) * startR,
            startY,
            Math.sin(startAngle) * startR
        );

        // Direction: roughly toward center with some randomness
        var dir = start.clone().negate().normalize();
        dir.x += (Math.random() - 0.5) * 0.5;
        dir.y += (Math.random() - 0.5) * 0.3;
        dir.z += (Math.random() - 0.5) * 0.5;
        dir.normalize();

        var length = 15 + Math.random() * 25;
        var end = start.clone().add(dir.clone().multiplyScalar(length));

        var geo = new THREE.BufferGeometry().setFromPoints([start, end]);
        var mat = new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
            linewidth: 2
        });
        var line = new THREE.Line(geo, mat);
        scene.add(line);

        var ss = {
            line: line,
            speed: 2 + Math.random() * 3,
            dir: dir,
            life: 1.0,
            decay: 0.015 + Math.random() * 0.02
        };
        shootingStars.push(ss);
    }

    var lastShootingStarTime = 0;

    function updateShootingStars(time) {
        // Spawn every 5-15 seconds
        if (time - lastShootingStarTime > 5000 + Math.random() * 10000) {
            spawnShootingStar();
            lastShootingStarTime = time;
        }

        for (var i = shootingStars.length - 1; i >= 0; i--) {
            var ss = shootingStars[i];
            ss.life -= ss.decay;
            ss.line.material.opacity = Math.max(0, ss.life);

            // Move along direction
            var positions = ss.line.geometry.attributes.position.array;
            for (var j = 0; j < 6; j += 3) {
                positions[j]     += ss.dir.x * ss.speed;
                positions[j + 1] += ss.dir.y * ss.speed;
                positions[j + 2] += ss.dir.z * ss.speed;
            }
            ss.line.geometry.attributes.position.needsUpdate = true;

            if (ss.life <= 0) {
                scene.remove(ss.line);
                ss.line.geometry.dispose();
                ss.line.material.dispose();
                shootingStars.splice(i, 1);
            }
        }
    }

    // =========================================================================
    // Mouse / Click interaction
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
                if (hoveredIndex !== intersects[0].index) {
                    hoveredIndex = intersects[0].index;
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

            var excerptStr = post.excerpt || '浩窚宇宙中，这颗星星上还没有留下摘录...';
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

            // Zoom to star
            var pos = points.geometry.attributes.position.array;
            var starPos = new THREE.Vector3(pos[selectedIndex*3], pos[selectedIndex*3+1], pos[selectedIndex*3+2]);

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

        var pos = points.geometry.attributes.position.array;
        var centerPos = new THREE.Vector3(pos[centerIndex*3], pos[centerIndex*3+1], pos[centerIndex*3+2]);

        var distances = [];
        for (var i = 0; i < postsData.length; i++) {
            if (i === centerIndex) continue;
            var p = new THREE.Vector3(pos[i*3], pos[i*3+1], pos[i*3+2]);
            distances.push({ index: i, pos: p, dist: centerPos.distanceTo(p) });
        }
        distances.sort(function(a, b) { return a.dist - b.dist; });

        var mat = new THREE.LineBasicMaterial({
            color: 0x66CCFF,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        distances.slice(0, maxConnections).forEach(function(n) {
            var geo = new THREE.BufferGeometry().setFromPoints([centerPos, n.pos]);
            connectionsGroup.add(new THREE.Line(geo, mat));
        });
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
                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(startPos)
                        .to({ x: 0, y: 0, z: 0 }, 1000 + Math.random() * 1000)
                        .delay(Math.random() * 1000)
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
            var rotSpeed = { speed: 0.05 };
            new TWEEN.Tween(rotSpeed)
                .to({ speed: 0.5 }, 2000)
                .onUpdate(function() {
                    if (blackHoleMesh) blackHoleMesh.children[0].rotation.z += rotSpeed.speed;
                })
                .start();
        }

        setTimeout(function() { window.toggleStarMap(); }, 2600);
    }

    function hideInfoPanel() {
        if (!infoPanel) return;
        infoPanel.style.transform = 'translateX(100%)';
        setTimeout(function() {
            if (infoPanel.style.transform === 'translateX(100%)') infoPanel.style.display = 'none';
        }, 400);
    }

    // =========================================================================
    // Animation loop
    // =========================================================================
    function animate(time) {
        if (!isAnimating) return;
        requestAnimationFrame(animate);

        if (typeof TWEEN !== 'undefined') TWEEN.update(time);

        var t = time * 0.001;

        // Star twinkling
        if (points && points.material.uniforms) {
            points.material.uniforms.uTime.value = t;
        }

        // Selection box spin
        if (selectionGroup && selectionGroup.visible && camera) {
            selectionGroup.lookAt(camera.position);
            selectionGroup.rotation.z += 0.01;
        }

        // Dust animation
        if (dustMaterial) {
            dustMaterial.uniforms.uTime.value = t;
        }

        // Black hole
        if (blackHoleMesh && !isWarping) {
            blackHoleMesh.children[0].rotation.z -= 0.005;
            blackHoleMesh.children[1].rotation.y += 0.002;
        }

        // Controls
        if (controls) {
            controls.update();
        } else if (points && selectedIndex === null) {
            points.rotation.y += 0.001;
        }

        // Dust rotation
        if (dustPoints && selectedIndex === null) {
            dustPoints.rotation.y += 0.0003;
            dustPoints.rotation.x += 0.0001;
            dustPoints.rotation.z += 0.0002;
        }

        // Shooting stars
        updateShootingStars(time);

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

        // Restore star positions
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
