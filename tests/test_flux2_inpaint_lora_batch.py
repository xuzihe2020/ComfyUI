from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "workflows"))

import run_flux2_inpaint_lora_batch as batch


class Flux2InpaintLoraBatchTests(unittest.TestCase):
    def test_discovers_image_mask_and_caption_without_treating_mask_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            mask = root / "sample_mask.png"
            caption = root / "sample.txt"
            image.write_bytes(b"image")
            mask.write_bytes(b"mask")
            caption.write_text("test prompt", encoding="utf-8")

            assets = batch.discover_assets(root, recursive=False, limit=None)

            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0].source, image)
            self.assertEqual(assets[0].mask, mask)
            self.assertEqual(assets[0].caption_path, caption)
            self.assertEqual(assets[0].caption, "test prompt")
            self.assertIsNone(assets[0].setup_error)

    def test_parses_comma_separated_denoise_values(self) -> None:
        settings = batch.parse_denoise_values("0.7, 0.80,0.9")
        self.assertEqual([item.text for item in settings], ["0.7", "0.8", "0.9"])
        self.assertEqual([item.value for item in settings], [0.7, 0.8, 0.9])
        with self.assertRaises(ValueError):
            batch.parse_denoise_values("0.8,0.8")
        with self.assertRaises(ValueError):
            batch.parse_denoise_values("1.2")

    def test_seeds_are_unique_across_images(self) -> None:
        values = iter([10, 10, 11, 12, 13])
        result = batch.generate_unique_seed_map(
            [Path("a.png"), Path("b.png")],
            2,
            randbelow=lambda _: next(values),
        )
        self.assertEqual(result, {"a.png": [10, 11], "b.png": [12, 13]})

    def test_jobs_group_by_lora_and_reuse_image_seeds_for_all_loras(self) -> None:
        assets = [
            batch.ImageAsset(Path("a.png"), Path("a.png")),
            batch.ImageAsset(Path("b.png"), Path("b.png")),
        ]
        loras = (
            batch.LoraSpec(Path("one.safetensors"), "one.safetensors", "one"),
            batch.LoraSpec(Path("two.safetensors"), "two.safetensors", "two"),
        )
        denoise = batch.parse_denoise_values("0.7,0.8")
        jobs = list(
            batch.iter_jobs(
                loras,
                assets,
                {"a.png": [101, 102], "b.png": [201, 202]},
                denoise,
            )
        )
        self.assertEqual(len(jobs), 16)
        self.assertEqual([job.lora.label for job in jobs[:8]], ["one"] * 8)
        self.assertEqual([job.lora.label for job in jobs[8:]], ["two"] * 8)
        one_a = [job.seed for job in jobs if job.lora.label == "one" and job.asset.relative_source == Path("a.png")]
        two_a = [job.seed for job in jobs if job.lora.label == "two" and job.asset.relative_source == Path("a.png")]
        self.assertEqual(one_a, [101, 101, 102, 102])
        self.assertEqual(two_a, one_a)

    def test_build_prompt_patches_all_runtime_dimensions_without_mutating_sidecar(self) -> None:
        prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "old", "clean_name": "old", "root_dir": "old"}},
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
            "17": {"class_type": "SplitSigmasDenoise", "inputs": {"denoise": 0.1}},
            "30": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old"}},
            "36": {"class_type": "LoadImageMask", "inputs": {"image": "old"}},
            "37": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "old"}},
        }
        bindings = [
            {"node_id": node_id, "endpoint": {"key": key}}
            for key, node_id in (
                ("input_image_01", "1"),
                ("input_mask_01", "36"),
                ("prompt_01", "10"),
                ("lora_01", "37"),
                ("denoise_01", "17"),
            )
        ]
        sidecar = {"prompt": prompt, "bindings": bindings}

        patched, save_id = batch.build_api_prompt(
            sidecar,
            staged_image="batch/image.png",
            staged_mask="batch/mask.png",
            source_image=Path("source.jpg"),
            caption="new prompt",
            lora_name="flux2\\test.safetensors",
            seed=123,
            denoise=0.8,
            save_prefix="batch/output",
        )

        self.assertEqual(save_id, "30")
        self.assertEqual(patched["1"]["inputs"]["image"], "batch/image.png")
        self.assertEqual(patched["36"]["inputs"]["image"], "batch/mask.png")
        self.assertEqual(patched["10"]["inputs"]["text"], "new prompt")
        self.assertEqual(patched["37"]["inputs"]["lora_name"], "flux2\\test.safetensors")
        self.assertEqual(patched["14"]["inputs"]["noise_seed"], 123)
        self.assertEqual(patched["17"]["inputs"]["denoise"], 0.8)
        self.assertEqual(prompt["14"]["inputs"]["noise_seed"], 1)

    def test_job_failure_does_not_stop_later_jobs(self) -> None:
        asset = batch.ImageAsset(Path("a.png"), Path("a.png"))
        lora = batch.LoraSpec(Path("one.safetensors"), "one.safetensors", "one")
        jobs = [
            batch.JobSpec(lora, asset, 1, 1, batch.DenoiseSetting("0.8", 0.8)),
            batch.JobSpec(lora, asset, 2, 2, batch.DenoiseSetting("0.8", 0.8)),
        ]
        visited: list[int] = []
        failures: list[int] = []

        def handler(index: int, _: batch.JobSpec) -> str:
            visited.append(index)
            if index == 1:
                raise RuntimeError("forced failure")
            return "succeeded"

        counts = batch.process_jobs(
            jobs,
            handler,
            lambda index, _job, _error: failures.append(index),
            fail_fast=False,
        )

        self.assertEqual(visited, [1, 2])
        self.assertEqual(failures, [1])
        self.assertEqual((counts.succeeded, counts.failed), (1, 1))


if __name__ == "__main__":
    unittest.main()
