# Visual Novel Image Generation Passes

This document describes the practical multi-pass image pipeline for visual
novel scene production. The goal is stable character identity, stable
environments, and controllable spatial layout, not a single magic workflow.

The core idea is:

```text
foundation layout
  -> environment/integration polish
  -> outfit/body refinement
  -> face/identity refinement
  -> hands/props/local repair
  -> upscale/final polish
```

Each pass should have a narrow job. If one pass tries to solve environment,
pose, outfit, face identity, lighting, and final detail at the same time, the
workflow becomes hard to control and hard to debug.

## 1. Production Target

For a VN scene, we usually need a small set of images that share:

- the same environment,
- the same character identity,
- compatible lighting and camera language,
- different positions and poses,
- enough consistency that they can belong to one scene.

For example, one cafe scene might need five foundation images:

- character standing near the counter,
- character sitting at a table,
- character leaning beside a window,
- character in a bust shot near the doorway,
- character turned away or walking through the room.

The first goal is not perfect face or clothing. The first goal is correct
spatial blocking: where the person is, how large they are, what pose they are
in, and how they relate to nearby objects.

## 2. Pass Summary

### Pass 1: Foundation Layout

Purpose:

- Put a person into the target environment at a controlled position.
- Establish rough body size, pose, camera distance, and scene composition.
- Preserve the environment outside the intended person area.
- Produce several candidate foundation images for the same scene.

This pass should answer:

- Is the person standing/sitting/bust/full-body?
- Where is the person in the room?
- Is the scale believable?
- Is the body facing the right direction?
- Does the person interact correctly with furniture or floor contact?

It does not need to fully solve:

- exact face identity,
- exact outfit fidelity,
- final hands,
- perfect lighting,
- final background cleanliness.

Current workflow:

```text
user/default/workflows/vn_foundation_environment_person_flux2_layout.json
```

### Pass 2: Environment And Integration Polish

Purpose:

- Clean up the environment after the person has been placed.
- Improve lighting, contact shadows, occlusion, and perspective integration.
- Preserve the basic person/environment relationship from the foundation pass.

Typical mask:

- background plus nearby interaction objects, if needed,
- or a soft border around the person where integration is weak.

Important detail:

Do not blindly mask the whole background if the character is already positioned
correctly. If the person is sitting in a chair or leaning on a table, the
chair/table/contact area may need to be included in the integration mask.

### Pass 3: Outfit / Body Refinement

Purpose:

- Bring clothing closer to the outfit reference.
- Fix body silhouette, sleeves, skirt, shoes, accessories, and pose details.
- Keep face and environment mostly stable.

Typical mask:

- clothing/body region,
- excluding face when possible,
- including hands only if the outfit interaction requires it.

This pass is where an outfit reference image is most useful. It should not be
expected to also solve face identity perfectly.

### Pass 4: Face / Identity Refinement

Purpose:

- Restore the intended character identity.
- Improve eyes, mouth, face shape, expression, and local hairline details.
- Preserve pose, outfit, camera, and background.

Typical mask:

- face area with a small margin,
- sometimes including ears or hairline,
- not the whole head/hair/neck unless the target is a hairstyle edit.

This is intentionally later than the foundation pass. The face pass works best
when the body, camera, and environment are already acceptable.

### Pass 5: Hands, Props, And Local Repair

Purpose:

- Fix hands, small accessories, object contact, props, signs, seams, or local
  artifacts.

Typical mask:

- only the broken local region,
- enough margin for blending,
- no unrelated areas.

This pass may be repeated. It is usually cheaper and more reliable to fix two
small regions separately than to mask half the image.

### Pass 6: Upscale / Final Polish

Purpose:

- Increase resolution.
- Restore fine texture and edge detail.
- Normalize color/contrast across scene images.

This pass should avoid changing composition or identity. If the final upscaler
changes the character too much, lower denoise or move the identity repair after
upscale.

## 3. Foundation Layout Pass In Detail

The foundation pass turns script text into concrete image geometry.

Script text is abstract:

```text
She waits near the cafe window, slightly turned away, nervous.
```

The foundation pass makes it concrete:

```text
The character is placed on the right third of the frame, bust-to-knee crop,
standing beside the window, body angled 30 degrees left, head turned toward
camera, window light from screen left, feet/floor contact hidden by crop.
```

This concrete blocking is the foundation for all later consistency work.

### 3.1 Inputs

The foundation workflow should use these inputs:

- Environment/canvas image: the actual room, street, cafe, bedroom, etc.
- Placement mask: where the person should be generated.
- Optional silhouette/rough mask: shaped person area for better pose/scale.
- Optional pose/skeleton image: visual guide for body pose.
- Person reference images: rough identity, body type, character style.
- Optional character LoRA: stronger character prior when available.
- Prompt: describes shot type, pose, camera distance, and interaction.

In the current workflow, the environment image is not just a reference. It is
the actual image being inpainted. This is important because VN environments
need to remain reusable and consistent.

### 3.2 Two Mask Input Modes

The current foundation workflow supports two manual placement styles.

#### Mode A: Shaped Silhouette Mask

Use this when you care about body silhouette and pose.

Process:

1. Load the environment image.
2. Load a silhouette or rough mask image.
3. Use `Compositor3` to move, scale, rotate, or stretch the silhouette over the
   environment.
4. Use `CompositorMasksOutputV3` to output the transformed full-canvas mask.
5. Feed that mask into `SetLatentNoiseMask`.

This is the best mode for:

- standing full-body placement,
- sitting pose placement,
- bust/body crop control,
- repeated positions across several images,
- “as smooth as Photoshop” manual layout work.

Mask source switch near the silhouette input:

- `1`: use the alpha mask from the loaded silhouette image.
- `2`: extract white areas from a white-on-black silhouette image.

Default is `2`, because white-on-black masks are easy to create and inspect.

#### Mode B: Quick Painted Region / Box Mask

Use this when speed matters more than pose silhouette.

Process:

1. Load the environment image.
2. Use `PreviewBridge` to paint or block out a rough mask over the image.
3. Set the final `ImageMaskSwitch` to input `2`.
4. Run the foundation inpaint.

This is best for:

- quick exploration,
- rough thumbnails,
- simple bust placement,
- testing prompt/camera ideas.

It is less precise than the silhouette path because a rectangular or rough mask
does not encode body shape.

### 3.3 Why The Mask Is The Main Control

For this pass, the mask is not only an inpaint region. It is a layout contract.

The mask tells the model:

- where the person is allowed to appear,
- how large the person should be,
- what rough silhouette the body should occupy,
- what environment areas must remain stable.

The prompt tells the model what to draw, but the mask tells it where and how
much space the person gets. For VN work, that spatial control is often more
important than adding more prompt words.

### 3.4 Pose / Skeleton Control

The current workflow includes pose/skeleton as a visual placement layer, not as
active ControlNet.

Reason:

- The exact Flux2-compatible OpenPose ControlNet node/model must be verified
  before wiring it into a preserved workflow.
- A fake or incompatible ControlNet path would make the workflow brittle.

Practical current approach:

- Use a silhouette or pose image as a guide layer in the compositor.
- Place it over the environment manually.
- Let the mask and prompt drive the foundation generation.

Future stronger approach:

- Add a verified Flux2-compatible pose ControlNet.
- Feed an OpenPose skeleton image into the ControlNet path.
- Keep the same manually placed silhouette mask as the inpaint/noise mask.

The mask and the skeleton should agree. If the skeleton says “standing full
body” but the mask is a small bust rectangle, the model receives conflicting
instructions.

### 3.5 Prompting The Foundation Pass

The foundation prompt should be concrete and spatial.

Good prompt shape:

```text
visual novel foundation image, insert the referenced woman into the masked
area of the provided cafe environment, standing full body near the window,
body angled slightly left, head turned toward camera, natural contact shadow,
matching warm cafe lighting, correct scale and perspective, preserve the room
outside the mask
```

Good negative prompt shape:

```text
wrong position, wrong scale, floating body, bad contact shadow, extra limbs,
deformed hands, duplicate person, changed environment outside mask, distorted
furniture, text, watermark, low quality
```

Avoid abstract-only prompts like:

```text
beautiful woman in cafe, cinematic, high quality
```

That does not define the spatial relationship.

### 3.6 What To Judge In Foundation Outputs

Accept a foundation image if:

- the person is in the right part of the room,
- the crop type is right,
- the body scale is believable,
- the pose is close enough,
- the environment outside the mask remains usable,
- the person can be refined later.

Reject a foundation image if:

- the body is in the wrong position,
- the scale is wrong,
- the pose is wrong,
- the person is floating,
- important furniture interaction is broken,
- the environment changed too much outside the intended area.

Do not reject only because:

- the face is not exact,
- the outfit is only approximate,
- hands need repair,
- lighting needs a later polish pass.

Those are later-pass problems.

## 4. Recommended Scene Workflow

For one VN scene/environment:

1. Choose or generate the clean environment plate.
2. Create five layout masks or silhouette placements for the needed shots.
3. Run the foundation workflow for each placement.
4. Pick the best foundation images based on spatial correctness.
5. Run environment/integration polish on selected images.
6. Run outfit/body refinement.
7. Run face/identity refinement.
8. Repair hands/props/local artifacts.
9. Upscale and final polish.

The important habit is to lock the composition early. Later passes should
improve the image, not keep changing the scene blocking.

## 5. Current Workflow Files

Foundation layout:

```text
user/default/workflows/vn_foundation_environment_person_flux2_layout.json
```

Face reference refinement:

```text
user/default/workflows/vn_face_reference_flux2_native_yolo_sam.json
user/default/workflows/vn_face_reference_flux2_native_yolo_sam_v2.json
```

Watermark/text repair examples:

```text
user/default/workflows/auto_text_watermark_fix_flux1_fill.json
user/default/workflows/auto_watermark_fix_flux1_yolo.json
```

## 6. Custom Nodes Relevant To This Plan

Manual placement:

```text
ComfyUI-enricos-nodes
```

Important node types:

```text
Compositor3
CompositorConfig3
CompositorMasksOutputV3
CompositorTools3
```

Manual painted mask:

```text
ComfyUI-Impact-Pack
PreviewBridge
ImageMaskSwitch
```

Mask cleanup:

```text
ImpactDilateMask
MaskBlur+
MaskFromColor+
```

Flux2 reference/edit path:

```text
ReferenceLatent
FluxKontextMultiReferenceLatentMethod
SetLatentNoiseMask
```

## 7. Design Rule

For VN production, do not ask one generation to solve everything.

Use the foundation pass to solve spatial truth first:

```text
who is where, doing what, at what scale, in what camera framing
```

Then use later passes to solve visual truth:

```text
identity, outfit, lighting, hands, props, final detail
```

This separation is what makes the pipeline debuggable.
