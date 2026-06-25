import { app } from "/scripts/app.js";

app.registerExtension({
    name: "flux.lookSelectors",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // --- ОБЩАЯ ФУНКЦИЯ ДЛЯ РАСЧЕТА РАЗМЕРА ---
        const setupDynamicSize = (node, previewNode, imgWidget) => {
            let currentAspectRatio = 1.0;

            previewNode.onload = function() {
                if (previewNode.naturalWidth && previewNode.naturalHeight) {
                    currentAspectRatio = previewNode.naturalHeight / previewNode.naturalWidth;
                }
                const targetHeight = node.size[0] * currentAspectRatio + getOtherWidgetsHeight(node);
                node.setSize([node.size[0], targetHeight]);
                app.graph.setDirtyCanvas(true, false);
            };

            const getOtherWidgetsHeight = (targetNode) => {
                let h = 0;
                for (const w of targetNode.widgets) {
                    if (w.name !== imgWidget.name) {
                         let wHeight = (w.computeSize ? w.computeSize()[1] : 30);
                         if (w.name === "extra_instruction") wHeight = 100;
                         h += wHeight + 4;
                    }
                }
                return h + 20;
            };

            imgWidget.computeSize = function(width) {
                return [width, width * currentAspectRatio];
            };

            const origOnResize = node.onResize;
            node.onResize = function(size) {
                const imgH = size[0] * currentAspectRatio;
                const otherH = getOtherWidgetsHeight(node);
                size[1] = imgH + otherH;
                if (origOnResize) origOnResize.call(this, size);
            };
        };

        // ==========================================
        // 1. ЛОГИКА ДЛЯ HAIRSTYLE
        // ==========================================
        if (nodeData.name === "FluxHairstyleSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                const node = this;

                const genderWidget = this.widgets.find(w => w.name === "gender");
                const hairstyleWidget = this.widgets.find(w => w.name === "hairstyle");
                
                const previewNode = document.createElement("img");
                previewNode.src = "";
                previewNode.style.width = "100%";
                previewNode.style.height = "auto"; 
                previewNode.style.objectFit = "contain";
                previewNode.style.marginTop = "10px";
                previewNode.style.border = "1px solid #555";
                previewNode.style.display = "block";
                previewNode.style.borderRadius = "4px";
                previewNode.style.background = "#222";

                const imgWidget = this.addDOMWidget("style_preview_img", "img", previewNode, { serialize: false });

                // Подключаем магию размера
                setupDynamicSize(node, previewNode, imgWidget);

                // Логика данных
                let hairstylesData = { Female: [], Male: [] };
                
                const loadJson = async (gender) => {
                    try {
                        const response = await fetch(`/extensions/ComfyUI-FluxLookSelector/hairstyles_${gender.toLowerCase()}.json`);
                        if(response.ok) return await response.json();
                    } catch(e) { console.error(e); }
                    return [];
                };

                (async () => {
                    const [femData, maleData] = await Promise.all([loadJson("Female"), loadJson("Male")]);
                    hairstylesData.Female = femData;
                    hairstylesData.Male = maleData;
                    updateHairstyleList("Female");
                    updatePreview();
                })();

                function updateHairstyleList(selectedGender) {
                    const items = hairstylesData[selectedGender] || [];
                    const names = items.map(i => i.name);
                    hairstyleWidget.options.values = names;
                    if (!names.includes(hairstyleWidget.value) && names.length > 0) {
                        hairstyleWidget.value = names[0];
                    }
                }

                const updatePreview = () => {
                    const gender = genderWidget.value.toLowerCase();
                    const styleName = hairstyleWidget.value;
                    const items = hairstylesData[genderWidget.value];
                    const item = items?.find(i => i.name === styleName);
                    if (item && item.thumbnail) {
                        previewNode.src = `/extensions/ComfyUI-FluxLookSelector/thumbnails/${gender}/${item.thumbnail}`;
                    } else { previewNode.src = ""; }
                };

                const origGenderCallback = genderWidget.callback;
                genderWidget.callback = function (value) {
                    updateHairstyleList(value);
                    updatePreview();
                    if (origGenderCallback) origGenderCallback.call(this, value);
                };

                const origStyleCallback = hairstyleWidget.callback;
                hairstyleWidget.callback = function (value) {
                    updatePreview();
                    if (origStyleCallback) origStyleCallback.call(this, value);
                };

                node.size[0] = 350; 
                return r;
            };
        }

        // ==========================================
        // 2. ЛОГИКА ДЛЯ BEARD
        // ==========================================
        else if (nodeData.name === "FluxBeardSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                const node = this;

                const beardWidget = this.widgets.find(w => w.name === "beard_style");
                
                const previewNode = document.createElement("img");
                previewNode.src = "";
                previewNode.style.width = "100%";
                previewNode.style.height = "auto";
                previewNode.style.objectFit = "contain";
                previewNode.style.marginTop = "10px";
                previewNode.style.border = "1px solid #555";
                previewNode.style.display = "block";
                previewNode.style.borderRadius = "4px";
                previewNode.style.background = "#222";

                const imgWidget = this.addDOMWidget("beard_preview_img", "img", previewNode, { serialize: false });

                // Подключаем магию размера
                setupDynamicSize(node, previewNode, imgWidget);

                // Логика данных
                let beardsData = [];
                
                fetch("/extensions/ComfyUI-FluxLookSelector/beards.json")
                    .then(response => response.ok ? response.json() : [])
                    .then(data => {
                        beardsData = data;
                        updatePreview();
                    })
                    .catch(e => console.error(e));

                const updatePreview = () => {
                    const item = beardsData.find(i => i.name === beardWidget.value);
                    if (item && item.thumbnail) {
                        previewNode.src = `/extensions/ComfyUI-FluxLookSelector/thumbnails/beards/${item.thumbnail}`;
                    } else { previewNode.src = ""; }
                };

                const origCallback = beardWidget.callback;
                beardWidget.callback = function (value) {
                    updatePreview();
                    if (origCallback) origCallback.call(this, value);
                };

                node.size[0] = 350; 
                return r;
            };
        }
    }
});