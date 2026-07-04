import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function imageLooksLikeMaskEditorTemp(value) {
    if (!value) {
        return false;
    }

    const normalized = String(value).replaceAll("\\", "/");
    const basename = normalized.split("/").pop() ?? "";
    return normalized.startsWith("clipspace/") || basename.startsWith("clipspace-painted-");
}

function setReadOnlyValue(widget, value) {
    const safeValue = value ?? "";
    widget._derivedDisplayValue = safeValue;
    widget.value = safeValue;
    widget.label = safeValue;

    if (widget.inputEl) {
        widget.inputEl.value = safeValue;
        widget.inputEl.disabled = true;
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = "0.55";
        widget.inputEl.style.cursor = "not-allowed";
    }
}

function ellipsizeText(ctx, text, maxWidth) {
    const value = String(text ?? "");
    if (ctx.measureText(value).width <= maxWidth) {
        return value;
    }

    const ellipsis = "...";
    let trimmed = value;
    while (trimmed.length > 0 && ctx.measureText(`${trimmed}${ellipsis}`).width > maxWidth) {
        trimmed = trimmed.slice(1);
    }
    return `${ellipsis}${trimmed}`;
}

function drawReadOnlyWidget(widget, ctx, node, width, y, height) {
    const margin = 15;
    const innerWidth = width - margin * 2;
    const labelWidth = Math.min(110, innerWidth * 0.38);
    const valueWidth = Math.max(20, innerWidth - labelWidth - 10);
    const centerY = y + height * 0.68;

    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = "#252525";
    ctx.strokeStyle = "#343434";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect?.(margin, y + 3, innerWidth, height - 6, 10);
    if (!ctx.roundRect) {
        ctx.rect(margin, y + 3, innerWidth, height - 6);
    }
    ctx.fill();
    ctx.stroke();

    ctx.globalAlpha = 1;
    ctx.font = `${Math.round(height * 0.55)}px Arial`;
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = "#777";
    ctx.fillText(widget.name, margin + 10, centerY);

    ctx.textAlign = "right";
    ctx.fillStyle = "#9a9a9a";
    ctx.fillText(ellipsizeText(ctx, widget.value, valueWidth), width - margin - 10, centerY);
    ctx.restore();
}

function makeReadOnly(widget) {
    if (!widget || widget._derivedDisplayReadOnly) {
        return;
    }

    widget.readonly = true;
    widget.disabled = true;
    widget.serialize = true;
    widget.options = {
        ...(widget.options ?? {}),
        disabled: true,
        readonly: true,
        readOnly: true,
        read_only: true,
    };

    widget.mouse = function () {
        setReadOnlyValue(widget, widget._derivedDisplayValue ?? widget.value ?? "");
        return true;
    };
    widget.draw = function (ctx, node, width, y, height) {
        drawReadOnlyWidget(widget, ctx, node, width, y, height);
    };

    const originalCallback = widget.callback;
    widget.callback = function () {
        setReadOnlyValue(widget, widget._derivedDisplayValue ?? widget.value ?? "");
        return originalCallback?.apply(this, arguments);
    };

    setReadOnlyValue(widget, widget.value ?? "");
    widget._derivedDisplayReadOnly = true;
}

function setStoredSource(node, cleanName, rootDir, sourcePath = undefined, imageValue = undefined) {
    node.properties ??= {};
    node.properties.load_image_source_clean_name = cleanName ?? "";
    node.properties.load_image_source_root_dir = rootDir ?? "";
    if (sourcePath !== undefined) {
        node.properties.load_image_source_path = sourcePath ?? "";
    }
    if (imageValue !== undefined) {
        node.properties.load_image_source_image = imageValue ?? "";
    }
}

function restoreStoredSource(node, cleanNameWidget, rootDirWidget) {
    const cleanName = node.properties?.load_image_source_clean_name;
    const rootDir = node.properties?.load_image_source_root_dir;

    if (!cleanName && !rootDir) {
        return false;
    }

    setReadOnlyValue(cleanNameWidget, cleanName);
    setReadOnlyValue(rootDirWidget, rootDir);
    return true;
}

function ensureWidget(node, name) {
    let widget = findWidget(node, name);
    if (widget) {
        return widget;
    }

    widget = node.addWidget("text", name, "", function () {
        setReadOnlyValue(widget, widget._derivedDisplayValue);
    });

    const imageIndex = node.widgets.findIndex((candidate) => candidate.name === "image");
    if (imageIndex >= 0) {
        node.widgets = node.widgets.filter((candidate) => candidate !== widget);
        node.widgets.splice(imageIndex + (name === "clean_name" ? 1 : 2), 0, widget);
    }

    return widget;
}

function isAbsolutePath(value) {
    const text = String(value ?? "");
    return /^[A-Za-z]:[\\/]/.test(text) || text.startsWith("\\\\") || text.startsWith("/");
}

function dirname(value) {
    const text = String(value ?? "");
    const index = Math.max(text.lastIndexOf("\\"), text.lastIndexOf("/"));
    return index >= 0 ? text.slice(0, index) : "";
}

function getStoredSourcePath(node, imageValue) {
    const sourceImage = node.properties?.load_image_source_image;
    const sourcePath = node.properties?.load_image_source_path;
    if (!sourcePath || sourceImage !== imageValue) {
        return "";
    }
    return sourcePath;
}

function setImageWidgetValue(node, imageName) {
    const imageWidget = findWidget(node, "image");
    if (!imageWidget || !imageName) {
        return;
    }

    imageWidget.value = imageName;
    imageWidget.label = imageName;

    preserveImageWidgetValue(imageWidget);

    imageWidget.callback?.(imageName);
    node._loadImageDerivedDisplayLastImage = imageName;
    scheduleDerivedUpdate(node, "local_file_picker");
    app.graph?.setDirtyCanvas(true, true);
}

function preserveImageWidgetValue(imageWidget) {
    const path = imageWidget?.value;
    if (!path) {
        return;
    }

    if (Array.isArray(imageWidget.options?.values) && !imageWidget.options.values.includes(path)) {
        imageWidget.options.values = [path, ...imageWidget.options.values];
    }
}

async function pickLocalImage(node, pickerWidget) {
    const imageWidget = findWidget(node, "image");
    const rootDirWidget = findWidget(node, "root_dir");
    const currentImage = imageWidget?.value ?? "";
    const sourcePath = getStoredSourcePath(node, currentImage);
    const initialDir = rootDirWidget?.value || (isAbsolutePath(sourcePath) ? dirname(sourcePath) : "");
    const previousValue = pickerWidget?.value;

    if (pickerWidget) {
        pickerWidget.value = "opening...";
    }

    try {
        const stripDoubleWidget = findWidget(node, "strip_double_underscore_suffix");
        const stripVersionWidget = findWidget(node, "strip_version_suffix");
        const params = new URLSearchParams({
            initial_dir: initialDir,
            strip_double_underscore_suffix: String(stripDoubleWidget?.value ?? true),
            strip_version_suffix: String(stripVersionWidget?.value ?? true),
        });
        const response = await api.fetchApi(`/local_file_picker/pick_load_image?${params.toString()}`, {
            cache: "no-store",
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (data.image) {
            setStoredSource(node, data.clean_name, data.root_dir, data.source_path, data.image);
            const cleanNameWidget = findWidget(node, "clean_name");
            const rootDirWidget = findWidget(node, "root_dir");
            if (cleanNameWidget) {
                setReadOnlyValue(cleanNameWidget, data.clean_name);
            }
            if (rootDirWidget) {
                setReadOnlyValue(rootDirWidget, data.root_dir);
            }
            setImageWidgetValue(node, data.image);
        }
    } catch (error) {
        console.warn("[LoadImage local file picker] Failed to pick local image", error);
    } finally {
        if (pickerWidget) {
            pickerWidget.value = previousValue ?? "choose file to upload";
        }
    }
}

function removeExtraLocalFilePickerButton(node) {
    const widget = findWidget(node, "choose local source");
    if (widget) {
        node.widgets = node.widgets?.filter((candidate) => candidate !== widget);
    }
}

function patchUploadButtonForLocalFilePicker(node) {
    removeExtraLocalFilePickerButton(node);

    let widget = findWidget(node, "upload");
    if (!widget) {
        widget = node.addWidget("button", "upload", "choose file to upload", () => {
            pickLocalImage(node, widget);
        });
    } else {
        widget.value = "choose file to upload";
        widget.callback = () => pickLocalImage(node, widget);
    }

    widget.serialize = false;
    widget.options = {
        ...(widget.options ?? {}),
        local_file_picker: true,
    };

    return widget;
}

function fitNodeToWidgets(node) {
    const computedSize = node.computeSize?.();
    if (computedSize) {
        node.setSize([
            Math.max(node.size[0], computedSize[0]),
            Math.max(node.size[1], computedSize[1]),
        ]);
    }
}

function scheduleDerivedUpdate(node, reason = "update") {
    clearTimeout(node._loadImageDerivedDisplayTimer);
    node._loadImageDerivedDisplayTimer = setTimeout(() => updateDerivedDisplay(node, reason), 80);
}

function wrapWidgetCallback(node, widget, reason) {
    if (!widget || widget._derivedDisplayWrapped) {
        return;
    }

    const originalCallback = widget.callback;
    widget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        scheduleDerivedUpdate(node, reason);
        return result;
    };
    widget._derivedDisplayWrapped = true;
}

async function updateDerivedDisplay(node, reason = "update") {
    const cleanNameWidget = findWidget(node, "clean_name");
    const rootDirWidget = findWidget(node, "root_dir");
    const imageWidget = findWidget(node, "image");

    if (!cleanNameWidget || !rootDirWidget || !imageWidget?.value) {
        if (cleanNameWidget) {
            setReadOnlyValue(cleanNameWidget, "");
        }
        if (rootDirWidget) {
            setReadOnlyValue(rootDirWidget, "");
        }
        app.graph?.setDirtyCanvas(true, false);
        return;
    }

    if (
        imageLooksLikeMaskEditorTemp(imageWidget.value)
        && (cleanNameWidget.value || rootDirWidget.value)
    ) {
        setReadOnlyValue(cleanNameWidget, cleanNameWidget.value);
        setReadOnlyValue(rootDirWidget, rootDirWidget.value);
        app.graph?.setDirtyCanvas(true, false);
        return;
    }

    if (imageLooksLikeMaskEditorTemp(imageWidget.value) && restoreStoredSource(node, cleanNameWidget, rootDirWidget)) {
        app.graph?.setDirtyCanvas(true, false);
        return;
    }

    const requestId = (node._loadImageDerivedDisplayRequestId ?? 0) + 1;
    node._loadImageDerivedDisplayRequestId = requestId;

    const stripDoubleWidget = findWidget(node, "strip_double_underscore_suffix");
    const stripVersionWidget = findWidget(node, "strip_version_suffix");

    const params = new URLSearchParams({
        image: imageWidget.value,
        strip_double_underscore_suffix: String(stripDoubleWidget?.value ?? true),
        strip_version_suffix: String(stripVersionWidget?.value ?? true),
    });
    const sourcePath = getStoredSourcePath(node, imageWidget.value);
    if (sourcePath) {
        params.set("source_path", sourcePath);
    }

    try {
        const response = await api.fetchApi(`/load_image/derived_display?${params.toString()}`, {
            cache: "no-store",
        });

        if (!response.ok || node._loadImageDerivedDisplayRequestId !== requestId) {
            return;
        }

        const data = await response.json();
        setReadOnlyValue(cleanNameWidget, data.clean_name);
        setReadOnlyValue(rootDirWidget, data.root_dir);
        setStoredSource(node, data.clean_name, data.root_dir, data.source_path, imageWidget.value);
        app.graph?.setDirtyCanvas(true, false);
    } catch (error) {
        console.warn("[LoadImage derived display] Failed to update derived fields", error);
    }
}

function setupLoadImageDerivedDisplay(node) {
    preserveImageWidgetValue(findWidget(node, "image"));
    patchUploadButtonForLocalFilePicker(node);
    const cleanNameWidget = ensureWidget(node, "clean_name");
    const rootDirWidget = ensureWidget(node, "root_dir");

    makeReadOnly(cleanNameWidget);
    makeReadOnly(rootDirWidget);
    if (!cleanNameWidget.value && !rootDirWidget.value) {
        restoreStoredSource(node, cleanNameWidget, rootDirWidget);
    } else {
        setStoredSource(node, cleanNameWidget.value, rootDirWidget.value);
    }
    fitNodeToWidgets(node);

    if (!node._loadImageDerivedDisplayReady) {
        wrapWidgetCallback(node, findWidget(node, "image"), "image");
        wrapWidgetCallback(node, findWidget(node, "strip_double_underscore_suffix"), "strip_double_underscore_suffix");
        wrapWidgetCallback(node, findWidget(node, "strip_version_suffix"), "strip_version_suffix");
        node._loadImageDerivedDisplayLastImage = findWidget(node, "image")?.value;
        node._loadImageDerivedDisplayInterval = setInterval(() => {
            const imageWidget = findWidget(node, "image");
            if (imageWidget?.value !== node._loadImageDerivedDisplayLastImage) {
                node._loadImageDerivedDisplayLastImage = imageWidget?.value;
                scheduleDerivedUpdate(node, "image");
            }
        }, 300);
        node._loadImageDerivedDisplayReady = true;
    }

    scheduleDerivedUpdate(node, "initial");
}

app.registerExtension({
    name: "comfy.loadImageDerivedDisplay",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadImage") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            setupLoadImageDerivedDisplay(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            setTimeout(() => setupLoadImageDerivedDisplay(this), 0);
            return result;
        };
    },
});
