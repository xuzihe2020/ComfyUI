# RunPod generation workflow

`flux2_txt2img_5_references.json` is the editable manual twin of
`aigc-infra/training/pipeline/generation.py`.

The workflow is standard Flux 2 text-to-image with five ordered reference
slots. All reference `LoadImage` → `VAEEncode` → `ReferenceLatent` triplets
are bypassed by default, so zero-reference generation works immediately.
For N references, enable complete triplets 1 through N and leave N+1 through
5 bypassed. The prompt's `Image 1`, `Image 2`, and subsequent guide lines must
describe the images loaded into those same numbered slots.

The RunPod batch runner creates its API graph directly. It includes only the
provided reference triplets, which is functionally equivalent to bypassing
the unused static slots in this workflow.
