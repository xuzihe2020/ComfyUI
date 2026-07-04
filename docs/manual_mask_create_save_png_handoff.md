# Manual Mask Creation + Auto Save PNG Handoff

## Goal

Create a small ComfyUI utility workflow for manually painting a mask on an input image, processing that mask, previewing it, and saving it as a normal PNG mask file.

The intended workflow is:

```text
Load image
Paint/edit mask with ComfyUI mask editor
Preview raw mask
Blur/process mask with tunable amount
Preview final mask
Convert MASK to IMAGE
Save final PNG mask automatically
```

## Required Save Path Behavior

The saved mask must go under a `masks` folder beside the image root that ComfyUI resolves for the selected image.

For the current common case where the image dropdown value is:

```text
char_01.jpg
```

ComfyUI resolves that as:

```text
C:\Users\Tony Xu\workspace\comfyui\input\char_01.jpg
```

Therefore the desired saved mask path is:

```text
C:\Users\Tony Xu\workspace\comfyui\input\masks\char_01_mask.png
```

If the image value is an absolute path such as:

```text
C:\Users\Tony Xu\workspace\Images\Inputs\tests\char_01.jpg
```

then the desired saved mask path is:

```text
C:\Users\Tony Xu\workspace\Images\Inputs\tests\masks\char_01_mask.png
```

## Filename Cleaning Rules

The derived clean name should come from the image filename stem.

Rules:

```text
char_01.jpg -> char_01
my_image_v01.png -> my_image
my_image__20260704_143000.png -> my_image
my_image_v01__anything.png -> my_image
```

Required configurable options:

```text
folder_name: masks
filename_suffix: _mask
extension: png
strip_double_underscore_suffix: true
strip_version_suffix: true
```

## Important Design Requirement

Do not depend on frontend JavaScript to compute or pass the actual save path.

The real save path must be computed in backend Python and passed through backend node outputs into the save node.

Frontend UI fields such as `clean_name` and `root_dir` are only display helpers. They can be blank, stale, cached, or fail to load. Saving must still work without them.

## Correct Backend Node Shape

A reusable backend node should load the image and derive save path values.

Inputs:

```text
image
folder_name
filename_suffix
extension
strip_double_underscore_suffix
strip_version_suffix
```

Outputs:

```text
IMAGE
MASK
clean_name
root_dir
output_path
filename_prefix
file_path
source_path
```

Backend logic:

```python
source_path = os.path.abspath(folder_paths.get_annotated_filepath(image))
root_dir = os.path.dirname(source_path)
output_path = os.path.join(root_dir, folder_name)
clean_name = clean_filename(os.path.basename(source_path))
filename_prefix = f"{clean_name}{filename_suffix}"
file_path = os.path.join(output_path, f"{filename_prefix}.{extension}")
```

For `char_01.jpg` with `folder_name=masks`, `filename_suffix=_mask`, `extension=png`, the backend outputs must be:

```text
root_dir = C:\Users\Tony Xu\workspace\comfyui\input
clean_name = char_01
output_path = C:\Users\Tony Xu\workspace\comfyui\input\masks
filename_prefix = char_01_mask
file_path = C:\Users\Tony Xu\workspace\comfyui\input\masks\char_01_mask.png
```

## Correct Workflow Wiring

The workflow should wire:

```text
Derived path node IMAGE -> Preview input image
Derived path node MASK -> raw mask preview
Derived path node MASK -> MaskBlur
PrimitiveInt -> MaskBlur amount
MaskBlur -> final mask preview
MaskBlur -> MaskToImage
MaskToImage IMAGE -> Image Save images
Derived path node output_path -> Image Save output_path
Derived path node filename_prefix -> Image Save filename_prefix
```

For WAS `Image Save`, use:

```text
overwrite_mode = prefix_as_filename
extension = png
```

With `overwrite_mode=prefix_as_filename`, WAS saves:

```text
{output_path}\{filename_prefix}.png
```

## Verification Checklist

After running the workflow, verify from `user/comfyui.log`.

Expected successful log pattern:

```text
got prompt
Image file saved to: C:\Users\Tony Xu\workspace\comfyui\input\masks\char_01_mask.png
Prompt executed in ...
```

If there is no new `got prompt`, the workflow did not execute.

If there is no new `Image file saved to: ...`, the save node did not run or failed.

If the file exists but its timestamp did not change, the latest run did not save/overwrite it.

Also verify filesystem:

```text
C:\Users\Tony Xu\workspace\comfyui\input\masks\char_01_mask.png
```

## Known Failed Approach

An attempted solution added a frontend extension:

```text
comfy_extras/web/image_derived_save_path/load_image_derived_save_path.js
```

That approach was not reliable. The UI fields stayed blank, and the frontend should not be trusted as the source of the save path anyway.

The correct fix should be backend-first: Python derives `output_path` and `filename_prefix`; those outputs are wired directly into the save node.

## Preservation Recommendation

Do not make this as an untracked local edit.

Best long-term path:

1. Create a small maintained custom node package outside ignored installed artifacts.
2. Include the backend derived-path node there.
3. Register it normally through `NODE_CLASS_MAPPINGS`.
4. Add it to the repo’s custom-node manifest or install process if needed.
5. Commit the workflow and node package.

If implemented inside ComfyUI core files, commit it to the user’s branch/fork before updating ComfyUI.
