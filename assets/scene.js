/* =========================================================================
   South Pole Station Foundation — Three.js 3D Scene
   Namespace: window.dash_clientside.settlement3d
   ========================================================================= */

(function () {
    "use strict";

    // Persistent app state — survives between Dash callback invocations
    const APP = {
        renderer: null,
        scene: null,
        camera: null,
        controls: null,
        columnMeshes: {},    // mp_id → { body: Mesh, cap: Mesh }
        beamMeshes: {},      // beam_id → floor-level beam Mesh
        gradeBeamMeshes: {}, // beam_id+"_gb" → grade-level beam Mesh (post-2017 only)
        refPlane: null,
        planes: { mean: null, all: null, podA: null, podB: null },
        meshToColumn: null,  // Map: body mesh → column data object
        meshToBeam: null,    // Map: beam mesh → { beam, isGrade }
        selectedDate: null,
        animFrameId: null,
        initialized: false,
        selectedMP: null,
        lastData: null,
        raycaster: new THREE.Raycaster(),
        mouse: new THREE.Vector2(),
        clickHandlerBound: false,
        hoverHandlerBound: false,
    };

    // -----------------------------------------------------------------------
    // Color utilities
    // -----------------------------------------------------------------------

    // Pile color: user-supplied min/max range → blue→cyan→yellow→red
    function valueToColor(value, minVal, maxVal) {
        const range = (maxVal - minVal) || 1;
        const t = Math.min(1.0, Math.max(0, (value - minVal) / range));
        const c = new THREE.Color();
        if (t < 0.33) {
            c.setRGB(0, 0.5 + t * 1.5, 1.0 - t * 3.0);
        } else if (t < 0.66) {
            const s = (t - 0.33) / 0.33;
            c.setRGB(s, 0.8, 0.1);
        } else {
            const s = (t - 0.66) / 0.34;
            c.setRGB(1.0, 0.8 * (1.0 - s), 0.0);
        }
        return c;
    }

    // Beam color: three-band threshold — green < 1 in, yellow 1–2 in, red ≥ 2 in.
    // Colors use zero blue component so the blue-tinted ambient light can't shift their hue.
    function beamDiffToColor(diffIn) {
        if (diffIn >= 3.0) return new THREE.Color(0xff0044); // bright magenta-red, > 3 in
        if (diffIn >= 2.0) return new THREE.Color(0xff2200); // red-orange,  2–3 in
        if (diffIn >= 1.0) return new THREE.Color(0xffcc00); // amber-yellow, 1–2 in
        return new THREE.Color(0x00cc00);                    // pure green,   < 1 in
    }

    // -----------------------------------------------------------------------
    // Plane fitting helpers
    // -----------------------------------------------------------------------

    function solve3x3(A, b) {
        const M = A.map((r, i) => [...r, b[i]]);
        for (let col = 0; col < 3; col++) {
            let maxRow = col;
            for (let row = col + 1; row < 3; row++)
                if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
            [M[col], M[maxRow]] = [M[maxRow], M[col]];
            if (Math.abs(M[col][col]) < 1e-12) return null;
            for (let row = col + 1; row < 3; row++) {
                const f = M[row][col] / M[col][col];
                for (let k = col; k <= 3; k++) M[row][k] -= f * M[col][k];
            }
        }
        const x = [0, 0, 0];
        for (let i = 2; i >= 0; i--) {
            x[i] = M[i][3];
            for (let j = i + 1; j < 3; j++) x[i] -= M[i][j] * x[j];
            x[i] /= M[i][i];
        }
        return x;
    }

    // Fit z = ax + by + c to an array of {x, y, z} points via least squares.
    function fitPlaneLSQ(pts) {
        const n = pts.length;
        if (n < 3) return null;
        let sx = 0, sy = 0, sz = 0, sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
        for (const p of pts) {
            sx += p.x; sy += p.y; sz += p.z;
            sxx += p.x * p.x; sxy += p.x * p.y; syy += p.y * p.y;
            sxz += p.x * p.z; syz += p.y * p.z;
        }
        const coeffs = solve3x3(
            [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]],
            [sxz, syz, sz]
        );
        if (!coeffs) return null;
        const [a, b, c] = coeffs;
        return { a, b, c, cx: sx / n, cy: sy / n };
    }

    // Position and orient a PlaneGeometry mesh to match z = ax + by + c, centred at (px, py).
    function applyFitPlane(mesh, fit, px, py) {
        const { a, b, c } = fit;
        mesh.position.set(px, py, a * px + b * py + c);
        mesh.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(-a, -b, 1).normalize()
        );
        mesh.visible = true;
    }

    // -----------------------------------------------------------------------
    // Column geometry factory
    // Column is a box: width × depth in XY, height in Z
    // -----------------------------------------------------------------------

    const COL_R = 1.5;           // pile radius (ft) — 36 inch diameter
    const VISUAL_COL_HEIGHT = 17; // ft — pile height

    // Rotation matrices for reorienting cylinder geometry
    const _zRotMat = new THREE.Matrix4().makeRotationX(Math.PI / 2);  // Y→Z (columns)
    const _xRotMat = new THREE.Matrix4().makeRotationZ(-Math.PI / 2); // Y→X (beam-axis tubes)

    function makeColumnMesh(color) {
        const geo = new THREE.CylinderGeometry(COL_R, COL_R, 1, 24);
        geo.applyMatrix4(_zRotMat); // height now runs along Z; scale.z controls pile length
        const mat = new THREE.MeshPhongMaterial({
            color: color,
            shininess: 40,
            specular: new THREE.Color(0x333333),
        });
        return new THREE.Mesh(geo, mat);
    }

    function makeCapMesh(color) {
        // Slightly wider disk representing the pile cap / grade beam bearing plate
        const geo = new THREE.CylinderGeometry(COL_R + 0.6, COL_R + 0.6, 1.5, 24);
        geo.applyMatrix4(_zRotMat);
        const mat = new THREE.MeshPhongMaterial({ color: color, shininess: 60 });
        return new THREE.Mesh(geo, mat);
    }

    function makeFloorBeamMesh() {
        const geo = new THREE.BoxGeometry(1, 2, 1.5);
        const mat = new THREE.MeshPhongMaterial({
            color: new THREE.Color(0x888888),
            shininess: 20,
            specular: new THREE.Color(0x111111),
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        return mesh;
    }

    function makeGradeBeamMesh() {
        const geo = new THREE.BoxGeometry(1, 1.5, 1);
        const mat = new THREE.MeshPhongMaterial({
            color: new THREE.Color(0x7b6f5a),
            shininess: 15,
            specular: new THREE.Color(0x111111),
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        return mesh;
    }

    function makePlaneMesh(w, h, color, opacity) {
        const geo = new THREE.PlaneGeometry(w, h);
        const mat = new THREE.MeshBasicMaterial({
            color, transparent: true, opacity,
            side: THREE.DoubleSide, depthWrite: false,
        });
        return new THREE.Mesh(geo, mat);
    }

    function makeInterPodMesh() {
        // Round tube — visually distinct structural type connecting the two pods
        const geo = new THREE.CylinderGeometry(0.4, 0.4, 1, 12);
        geo.applyMatrix4(_xRotMat); // height along X so scale.x stretches it
        const mat = new THREE.MeshPhongMaterial({
            color: new THREE.Color(0xb0bec5),
            shininess: 80,
            specular: new THREE.Color(0x607d8b),
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        return mesh;
    }

    // -----------------------------------------------------------------------
    // Scene initialisation
    // -----------------------------------------------------------------------

    function initScene() {
        const container = document.getElementById("three-canvas-container");
        if (!container || container.dataset.threeInit === "1") return;
        container.dataset.threeInit = "1";

        const W = container.clientWidth || 800;
        const H = container.clientHeight || 480;

        // Renderer
        APP.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        APP.renderer.setPixelRatio(window.devicePixelRatio);
        APP.renderer.setSize(W, H);
        APP.renderer.shadowMap.enabled = true;
        APP.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(APP.renderer.domElement);

        // Scene
        APP.scene = new THREE.Scene();
        APP.scene.background = new THREE.Color(0x0d1117);
        APP.scene.fog = new THREE.FogExp2(0x0d1117, 0.0008);

        // Camera — looking at foundation center from above-angle
        APP.camera = new THREE.PerspectiveCamera(45, W / H, 0.5, 5000);
        APP.camera.position.set(200, -280, 260);
        APP.camera.up.set(0, 0, 1);

        // Orbit controls
        APP.controls = new THREE.OrbitControls(APP.camera, APP.renderer.domElement);
        APP.controls.target.set(200, 60, 0);
        APP.controls.enableDamping = true;
        APP.controls.dampingFactor = 0.08;
        APP.controls.screenSpacePanning = false;
        APP.controls.maxPolarAngle = Math.PI * 0.85;
        APP.controls.update();

        // Lights
        const ambient = new THREE.AmbientLight(0x404060, 2.5);
        APP.scene.add(ambient);

        const sun = new THREE.DirectionalLight(0xfff8f0, 3.5);
        sun.position.set(300, -200, 400);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        sun.shadow.camera.near = 1;
        sun.shadow.camera.far = 2000;
        sun.shadow.camera.left = -500;
        sun.shadow.camera.right = 500;
        sun.shadow.camera.top = 400;
        sun.shadow.camera.bottom = -200;
        APP.scene.add(sun);

        const fill = new THREE.DirectionalLight(0x8090c0, 1.2);
        fill.position.set(-200, 300, 100);
        APP.scene.add(fill);

        // Grid reference plane
        const gridGeo = new THREE.PlaneGeometry(600, 300, 30, 15);
        const gridMat = new THREE.MeshBasicMaterial({
            color: 0x1a2233,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
        });
        APP.refPlane = new THREE.Mesh(gridGeo, gridMat);
        APP.refPlane.position.set(200, 60, 0);
        APP.refPlane.receiveShadow = true;
        APP.scene.add(APP.refPlane);

        // Grid helper for orientation
        const gridHelper = new THREE.GridHelper(600, 30, 0x223344, 0x1a2a38);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.set(200, 60, -0.1);
        APP.scene.add(gridHelper);

        // Axes helper (small, in corner)
        const axes = new THREE.AxesHelper(20);
        axes.position.set(-20, -20, 0);
        APP.scene.add(axes);

        // Resize handler
        window.addEventListener("resize", () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            APP.renderer.setSize(w, h);
            APP.camera.aspect = w / h;
            APP.camera.updateProjectionMatrix();
        });

        // Click + hover handlers
        if (!APP.clickHandlerBound) {
            APP.renderer.domElement.addEventListener("click", onSceneClick);
            APP.clickHandlerBound = true;
        }
        if (!APP.hoverHandlerBound) {
            APP.renderer.domElement.addEventListener("mousemove", onSceneHover);
            APP.renderer.domElement.addEventListener("mouseleave", onSceneLeave);
            APP.hoverHandlerBound = true;
        }

        // Animation loop
        function animate() {
            APP.animFrameId = requestAnimationFrame(animate);
            APP.controls.update();
            APP.renderer.render(APP.scene, APP.camera);
        }
        animate();

        APP.initialized = true;
    }

    // -----------------------------------------------------------------------
    // Click / selection
    // -----------------------------------------------------------------------

    function onSceneClick(event) {
        const canvas = APP.renderer.domElement;
        const rect = canvas.getBoundingClientRect();
        APP.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        APP.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        APP.raycaster.setFromCamera(APP.mouse, APP.camera);
        const meshes = Object.values(APP.columnMeshes).map(c => c.body).filter(Boolean);
        const hits = APP.raycaster.intersectObjects(meshes);

        if (hits.length > 0) {
            const hit = hits[0].object;
            APP.selectedMP = hit.userData.mp_id || null;
            window._spsSelectedMP = APP.selectedMP;
            // Highlight selected
            Object.entries(APP.columnMeshes).forEach(([id, parts]) => {
                if (parts.body) {
                    parts.body.material.emissive.setHex(id === APP.selectedMP ? 0x334455 : 0x000000);
                    parts.body.material.emissiveIntensity = id === APP.selectedMP ? 1 : 0;
                }
            });
        }
    }

    function onSceneHover(event) {
        if (!APP.initialized || !APP.lastData || !APP.selectedDate) return;

        const canvas = APP.renderer.domElement;
        const rect = canvas.getBoundingClientRect();
        const mx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const my = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        APP.raycaster.setFromCamera({ x: mx, y: my }, APP.camera);

        const tip = document.getElementById("three-tooltip");
        if (!tip) return;

        // Raycast columns first
        const colMeshes = Object.values(APP.columnMeshes).map(c => c.body).filter(Boolean);
        const colHits = APP.raycaster.intersectObjects(colMeshes);
        if (colHits.length > 0) {
            const col = APP.meshToColumn && APP.meshToColumn.get(colHits[0].object);
            if (col) {
                const s  = col.settlements[APP.selectedDate];
                const r  = col.settlement_rates[APP.selectedDate];
                const sh = col.shim_inches && col.shim_inches[APP.selectedDate];
                const sText  = s  != null ? s.toFixed(2)  + " in"     : "—";
                const rText  = r  != null ? r.toFixed(3)  + " in/yr"  : "—";
                const shText = sh != null ? sh.toFixed(2) + " in"     : "—";
                tip.innerHTML = `<b>${col.id}</b> &nbsp;Pod ${col.pod}<br>` +
                    `Settlement: ${sText}<br>Rate: ${rText}<br>Shim pack: ${shText}`;
                tip.style.left = (event.clientX - rect.left + 14) + "px";
                tip.style.top  = (event.clientY - rect.top  - 10) + "px";
                tip.style.display = "block";
                return;
            }
        }

        // Raycast beams (floor + grade)
        const beamMeshList = [
            ...Object.values(APP.beamMeshes),
            ...Object.values(APP.gradeBeamMeshes),
        ].filter(m => m.visible);
        const beamHits = APP.raycaster.intersectObjects(beamMeshList);
        if (beamHits.length > 0) {
            const entry = APP.meshToBeam && APP.meshToBeam.get(beamHits[0].object);
            if (entry) {
                const { beam, isGrade } = entry;
                const fd  = (beam.floor_diffs      || {})[APP.selectedDate];
                const gbd = (beam.grade_beam_diffs  || {})[APP.selectedDate];
                const diffLabel = isGrade ? "Grade beam diff" : "Floor diff";
                const diffVal   = isGrade ? gbd : fd;
                let html = `<b>${beam.id}</b><br>${beam.start_id} ↔ ${beam.end_id}`;
                if (diffVal != null) html += `<br>${diffLabel}: ${diffVal.toFixed(2)} in`;
                if (!isGrade && gbd != null) html += `<br>Grade beam diff: ${gbd.toFixed(2)} in`;
                tip.innerHTML = html;
                tip.style.left = (event.clientX - rect.left + 14) + "px";
                tip.style.top  = (event.clientY - rect.top  - 10) + "px";
                tip.style.display = "block";
                return;
            }
        }

        tip.style.display = "none";
    }

    function onSceneLeave() {
        const tip = document.getElementById("three-tooltip");
        if (tip) tip.style.display = "none";
    }

    // -----------------------------------------------------------------------
    // Scene update — called on each Dash callback
    // -----------------------------------------------------------------------

    function updateScene(data, dateIdx, metric, exaggeration, viewMode, colorMin, colorMax, planeModes, datumParam, beamColorMode) {
        if (!data) return window._spsSelectedMP || null;

        // Lazy init
        if (!APP.initialized) {
            initScene();
            if (!APP.initialized) return null;
        }

        const allDates = data.survey_dates.concat(data.proj_dates);
        const selectedDate = allDates[dateIdx];
        const isProj = data.proj_dates.includes(selectedDate);
        const cMin = colorMin ?? 0;
        const cMax = colorMax ?? (metric === "rate"
            ? (data.stats.max_rate_in_yr || 1)
            : (data.stats.max_settlement_in || 1));

        // ---- Compute mode-specific Z reference from non-null columns only ----
        const exag = exaggeration || 20;
        let minS = 0, meanGB = 0, datumGB = 0;
        if (viewMode === "bowl") {
            const sVals = data.columns
                .map(c => c.settlements[selectedDate] ?? (isProj ? c.proj_settlements[selectedDate] : null))
                .filter(v => v != null);
            minS = sVals.length > 0 ? Math.min(...sVals) : 0;
        } else if (viewMode === "elevation") {
            // Mean-referenced: zero = current average, so differential movement is visible
            const gbVals = data.columns
                .map(c => c.grade_beam_elevations[selectedDate])
                .filter(v => v != null);
            meanGB = gbVals.length > 0 ? gbVals.reduce((a, b) => a + b, 0) / gbVals.length : 0;
        } else { // "fixed" datum
            if (datumParam != null && !isNaN(datumParam)) {
                datumGB = parseFloat(datumParam);
            } else {
                // Input not yet populated — compute from earliest floor date
                const fd = (data.floor_dates || [])[0];
                if (fd) {
                    const gbVals = data.columns.map(c => c.grade_beam_elevations[fd]).filter(v => v != null);
                    datumGB = gbVals.length > 0 ? gbVals.reduce((a, b) => a + b, 0) / gbVals.length : 0;
                }
            }
        }

        // Track which columns are rendered at this date (for beam culling)
        const visibleColIds = new Set();

        // Rebuild hover lookup maps fresh each update
        APP.meshToColumn = new Map();
        APP.meshToBeam   = new Map();

        // ---- Update / create columns ----
        data.columns.forEach(col => {
            const rawSettlement = col.settlements[selectedDate]
                ?? (isProj ? col.proj_settlements[selectedDate] : null);
            const rate = col.settlement_rates[selectedDate] ?? 0;
            const gbElev = col.grade_beam_elevations[selectedDate];

            // Create mesh on first encounter
            if (!APP.columnMeshes[col.id]) {
                const body = makeColumnMesh(new THREE.Color(0x4fc3f7));
                body.userData.mp_id = col.id;
                body.castShadow = true;
                body.receiveShadow = true;
                const cap = makeCapMesh(new THREE.Color(0x4fc3f7));
                cap.userData.mp_id = col.id;
                APP.scene.add(body);
                APP.scene.add(cap);
                APP.columnMeshes[col.id] = { body, cap };
            }
            const { body, cap } = APP.columnMeshes[col.id];

            // Hide columns not yet installed at this date.
            // In elevation mode, fall back to z=0 if grade beam elev is unavailable but
            // the MP does have settlement data (e.g. pre-2017 MPs or missing shim reference).
            const notInstalled = rawSettlement == null && gbElev == null;
            if (notInstalled) {
                body.visible = false;
                cap.visible = false;
                return;
            }
            body.visible = true;
            cap.visible = true;
            visibleColIds.add(col.id);
            APP.meshToColumn.set(body, col);

            const settlement = rawSettlement ?? 0;

            // Z position
            let zBottom;
            if (viewMode === "bowl") {
                zBottom = -((settlement - minS) / 12) * exag;
            } else if (viewMode === "elevation") {
                zBottom = gbElev != null ? (gbElev - meanGB) * exag : 0;
            } else { // fixed datum
                zBottom = gbElev != null ? (gbElev - datumGB) * exag : 0;
            }
            const columnHeight = VISUAL_COL_HEIGHT;

            let color;
            if (metric === "settlement") {
                color = valueToColor(settlement, cMin, cMax);
            } else if (metric === "rate") {
                color = valueToColor(Math.abs(rate), cMin, cMax);
            } else {
                color = new THREE.Color(0x78909c); // "none" — neutral steel blue-gray
            }

            body.scale.set(1, 1, columnHeight);
            body.position.set(col.x, col.y, zBottom + columnHeight / 2);
            body.material.color.copy(color);
            cap.position.set(col.x, col.y, zBottom + columnHeight + 0.75);
            cap.material.color.copy(color);

            const isSelected = col.id === APP.selectedMP;
            body.material.emissive.setHex(isSelected ? 0x334455 : 0x000000);
            body.material.emissiveIntensity = isSelected ? 1 : 0;
        });

        // Shared helper: position, scale and orient a pre-built beam mesh between two 3-D points
        function placeBeam(mesh, x1, y1, z1, x2, y2, z2) {
            const dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
            const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (len < 0.01) { mesh.visible = false; return; }
            mesh.position.set((x1+x2)/2, (y1+y2)/2, (z1+z2)/2);
            mesh.scale.set(len, 1, 1);
            mesh.quaternion.setFromUnitVectors(
                new THREE.Vector3(1,0,0),
                new THREE.Vector3(dx/len, dy/len, dz/len));
        }

        // ---- Floor-level beams (top of columns) ----
        data.beams.forEach(beam => {
            if (beam.start_x == null || beam.end_x == null) return;

            if (!APP.beamMeshes[beam.id]) {
                const mesh = beam.is_inter_pod ? makeInterPodMesh() : makeFloorBeamMesh();
                APP.scene.add(mesh);
                APP.beamMeshes[beam.id] = mesh;
            }
            const mesh = APP.beamMeshes[beam.id];

            if (!visibleColIds.has(beam.start_id) || !visibleColIds.has(beam.end_id)) {
                mesh.visible = false; return;
            }
            mesh.visible = true;
            APP.meshToBeam.set(mesh, { beam, isGrade: false });

            const sCol = data.columns.find(c => c.id === beam.start_id);
            const eCol = data.columns.find(c => c.id === beam.end_id);
            const sS = sCol.settlements[selectedDate] ?? (isProj ? sCol.proj_settlements[selectedDate] : null) ?? 0;
            const eS = eCol.settlements[selectedDate] ?? (isProj ? eCol.proj_settlements[selectedDate] : null) ?? 0;

            const gbS = sCol.grade_beam_elevations[selectedDate], gbEE = eCol.grade_beam_elevations[selectedDate];
            let zS, zE;
            if (viewMode === "bowl") {
                zS = -((sS - minS) / 12) * exag + VISUAL_COL_HEIGHT;
                zE = -((eS - minS) / 12) * exag + VISUAL_COL_HEIGHT;
            } else if (viewMode === "elevation") {
                zS = (gbS != null ? (gbS - meanGB) * exag : 0) + VISUAL_COL_HEIGHT;
                zE = (gbEE != null ? (gbEE - meanGB) * exag : 0) + VISUAL_COL_HEIGHT;
            } else { // fixed datum
                zS = (gbS != null ? (gbS - datumGB) * exag : 0) + VISUAL_COL_HEIGHT;
                zE = (gbEE != null ? (gbEE - datumGB) * exag : 0) + VISUAL_COL_HEIGHT;
            }

            if (!beam.is_inter_pod) {
                if (beamColorMode === "gray") {
                    mesh.material.color.setHex(0x607080);
                } else {
                    const diff = beam.floor_diffs[selectedDate] ?? 0;
                    mesh.material.color.copy(beamDiffToColor(diff));
                    mesh.material.emissive.setHex(diff >= 3.0 ? 0x660011 : diff >= 2.0 ? 0x330000 : 0x000000);
                }
            }

            placeBeam(mesh, beam.start_x, beam.start_y, zS, beam.end_x, beam.end_y, zE);
        });

        // ---- Grade-level beams (bottom of columns) ----
        const showGradeBeams = true;

        data.beams.forEach(beam => {
            if (beam.start_x == null || beam.end_x == null) return;
            if (beam.is_inter_pod) return; // inter-pod connections are floor-level only
            const gbId = beam.id + "_gb";

            if (!APP.gradeBeamMeshes[gbId]) {
                const mesh = beam.is_inter_pod ? makeInterPodMesh() : makeGradeBeamMesh();
                APP.scene.add(mesh);
                APP.gradeBeamMeshes[gbId] = mesh;
            }
            const mesh = APP.gradeBeamMeshes[gbId];

            if (!showGradeBeams || !visibleColIds.has(beam.start_id) || !visibleColIds.has(beam.end_id)) {
                mesh.visible = false; return;
            }
            mesh.visible = true;
            APP.meshToBeam.set(mesh, { beam, isGrade: true });

            const sCol = data.columns.find(c => c.id === beam.start_id);
            const eCol = data.columns.find(c => c.id === beam.end_id);
            const sS = sCol.settlements[selectedDate] ?? (isProj ? sCol.proj_settlements[selectedDate] : null) ?? 0;
            const eS = eCol.settlements[selectedDate] ?? (isProj ? eCol.proj_settlements[selectedDate] : null) ?? 0;

            const gbGS = sCol.grade_beam_elevations[selectedDate], gbGE = eCol.grade_beam_elevations[selectedDate];
            let zGS, zGE;
            if (viewMode === "bowl") {
                zGS = -((sS - minS) / 12) * exag;
                zGE = -((eS - minS) / 12) * exag;
            } else if (viewMode === "elevation") {
                zGS = gbGS != null ? (gbGS - meanGB) * exag : 0;
                zGE = gbGE != null ? (gbGE - meanGB) * exag : 0;
            } else { // fixed datum
                zGS = gbGS != null ? (gbGS - datumGB) * exag : 0;
                zGE = gbGE != null ? (gbGE - datumGB) * exag : 0;
            }

            if (!beam.is_inter_pod) {
                if (beamColorMode === "gray") {
                    mesh.material.color.setHex(0x607080);
                } else {
                    const diff = (beam.grade_beam_diffs || {})[selectedDate] ?? 0;
                    mesh.material.color.copy(beamDiffToColor(diff));
                    mesh.material.emissive.setHex(diff >= 3.0 ? 0x660011 : diff >= 2.0 ? 0x330000 : 0x000000);
                }
            }

            placeBeam(mesh, beam.start_x, beam.start_y, zGS, beam.end_x, beam.end_y, zGE);
        });

        // ---- Reference / fit planes ----
        const activePlanes = planeModes || [];

        // Build column-top {x, y, z, pod} for all currently visible columns
        const colTops = data.columns
            .filter(c => visibleColIds.has(c.id))
            .map(c => {
                const gbE = c.grade_beam_elevations[selectedDate];
                const rawS = c.settlements[selectedDate] ?? (isProj ? c.proj_settlements[selectedDate] : null);
                const s = rawS ?? 0;
                let z;
                if (viewMode === "bowl")           z = -((s - minS) / 12) * exag + VISUAL_COL_HEIGHT;
                else if (viewMode === "elevation") z = (gbE != null ? (gbE - meanGB)  * exag : 0) + VISUAL_COL_HEIGHT;
                else                               z = (gbE != null ? (gbE - datumGB) * exag : 0) + VISUAL_COL_HEIGHT;
                return { x: c.x, y: c.y, z, pod: c.pod };
            });

        // Mean plane — flat horizontal at the mean column-top Z
        if (!APP.planes.mean) { APP.planes.mean = makePlaneMesh(440, 160, 0xffffff, 0.10); APP.scene.add(APP.planes.mean); }
        APP.planes.mean.visible = activePlanes.includes("mean") && colTops.length > 0;
        if (APP.planes.mean.visible) {
            const mz = colTops.reduce((a, p) => a + p.z, 0) / colTops.length;
            APP.planes.mean.position.set(200, 65, mz);
            APP.planes.mean.quaternion.identity();
        }

        // Best-fit plane — all visible columns
        if (!APP.planes.all) { APP.planes.all = makePlaneMesh(440, 160, 0x4fc3f7, 0.12); APP.scene.add(APP.planes.all); }
        if (activePlanes.includes("all") && colTops.length >= 3) {
            const fit = fitPlaneLSQ(colTops);
            if (fit) applyFitPlane(APP.planes.all, fit, 200, 65);
            else APP.planes.all.visible = false;
        } else { APP.planes.all.visible = false; }

        // Best-fit plane — Pod A
        if (!APP.planes.podA) { APP.planes.podA = makePlaneMesh(240, 160, 0xffb74d, 0.14); APP.scene.add(APP.planes.podA); }
        const podAPts = colTops.filter(p => p.pod === 'A');
        if (activePlanes.includes("podA") && podAPts.length >= 3) {
            const fit = fitPlaneLSQ(podAPts);
            const cx = podAPts.reduce((s, p) => s + p.x, 0) / podAPts.length;
            const cy = podAPts.reduce((s, p) => s + p.y, 0) / podAPts.length;
            if (fit) applyFitPlane(APP.planes.podA, fit, cx, cy);
            else APP.planes.podA.visible = false;
        } else { APP.planes.podA.visible = false; }

        // Best-fit plane — Pod B
        if (!APP.planes.podB) { APP.planes.podB = makePlaneMesh(240, 160, 0xf06292, 0.14); APP.scene.add(APP.planes.podB); }
        const podBPts = colTops.filter(p => p.pod === 'B');
        if (activePlanes.includes("podB") && podBPts.length >= 3) {
            const fit = fitPlaneLSQ(podBPts);
            const cx = podBPts.reduce((s, p) => s + p.x, 0) / podBPts.length;
            const cy = podBPts.reduce((s, p) => s + p.y, 0) / podBPts.length;
            if (fit) applyFitPlane(APP.planes.podB, fit, cx, cy);
            else APP.planes.podB.visible = false;
        } else { APP.planes.podB.visible = false; }

        // Move reference plane to z=0 (bowl) or at datum
        if (APP.refPlane) {
            APP.refPlane.position.z = viewMode === "bowl" ? 0.0 : -2;
        }

        APP.selectedDate = selectedDate;
        APP.lastData = data;
        window._spsSelectedMP = APP.selectedMP;
        return APP.selectedMP;
    }

    let _lastReturnedMP = undefined;
    function getSelectedMP(trigger) {
        const current = window._spsSelectedMP || null;
        if (current === _lastReturnedMP) return window.dash_clientside.no_update;
        _lastReturnedMP = current;
        return current;
    }

    // -----------------------------------------------------------------------
    // Register as Dash clientside namespace
    // -----------------------------------------------------------------------

    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.settlement3d = {
        updateScene: updateScene,
        getSelectedMP: getSelectedMP,
    };

    // Auto-init on DOM ready (handles initial page load before any callback fires)
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(initScene, 300);
        });
    } else {
        setTimeout(initScene, 300);
    }

})();
