from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO_ROOT / "scripts" / "install_custom_nodes.py"
SPEC = importlib.util.spec_from_file_location(
    "install_custom_nodes_for_test",
    INSTALLER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def integration_manifest() -> dict:
    return {
        "frontend": {
            "name": "Pinned frontend",
            "folder": "ComfyUI_frontend",
            "repo": "https://example.invalid/frontend.git",
            "ref": "f" * 40,
            "version": "1.49.0",
            "package_manager": "pnpm@11.13.1",
            "build_script": "build",
            "dist": "dist",
            "required_paths": ["package.json", "pnpm-lock.yaml"],
            "required_dist_paths": ["index.html"],
        },
        "nodes": [
            {
                "name": "comfyui-editor-bridge",
                "folder": "comfyui-editor-bridge",
                "repo": "https://example.invalid/bridge.git",
                "ref": "b" * 40,
                "install_method": "git-clone",
                "require_local_checkout": True,
                "required_paths": [
                    "__init__.py",
                    "editor_bridge/routes.py",
                ],
            }
        ],
    }


class EditorIntegrationManifestTests(unittest.TestCase):
    def test_real_manifest_pins_frontend_and_bridge(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / "custom_nodes.manifest.json").read_text(encoding="utf-8")
        )
        frontend = manifest["frontend"]
        self.assertEqual(
            frontend["repo"],
            "https://github.com/xuzihe2020/ComfyUI_frontend.git",
        )
        self.assertEqual(frontend["version"], "1.49.0")
        self.assertEqual(frontend["node_engine"], ">=25 <26")
        self.assertRegex(frontend["ref"], r"^[0-9a-f]{40}$")

        bridge = next(
            node
            for node in manifest["nodes"]
            if node["folder"] == "comfyui-editor-bridge"
        )
        self.assertEqual(bridge["install_method"], "git-clone")
        self.assertTrue(bridge["require_local_checkout"])
        self.assertRegex(bridge["ref"], r"^[0-9a-f]{40}$")

    def test_runpod_manifest_excludes_frontend_and_bridge(self) -> None:
        manifest = integration_manifest()
        filtered = installer.without_editor_integration(manifest)

        self.assertNotIn("frontend", filtered)
        self.assertEqual(filtered["nodes"], [])
        self.assertIn("frontend", manifest)
        self.assertEqual(len(manifest["nodes"]), 1)

    def test_run_script_converges_manifest_before_startup(self) -> None:
        run_script = (REPO_ROOT / "run_comfyui.bat").read_text(encoding="utf-8")
        installer_command = (
            r'".venv\Scripts\python.exe" '
            r'"scripts\install_custom_nodes.py"'
        )
        self.assertIn(installer_command, run_script)
        self.assertLess(
            run_script.index(installer_command),
            run_script.index(r'".venv\Scripts\python.exe" "main.py"'),
        )
        self.assertIn("if errorlevel 1 exit /b 1", run_script)
        self.assertIn(r"ComfyUI_frontend\dist", run_script)
        self.assertNotIn(r"tools\ComfyUI_frontend", run_script)
        self.assertNotIn(r"..\ComfyUI_frontend", run_script)
        self.assertNotIn("COMFYUI_EDITOR_BRIDGE_SOURCE", run_script)
        self.assertNotIn("using the packaged ComfyUI frontend", run_script.lower())


class EditorIntegrationInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.custom_nodes = self.root / "custom_nodes"
        self.tools.mkdir()
        self.custom_nodes.mkdir()
        self.manifest = integration_manifest()

        self.tools_patch = mock.patch.object(installer, "TOOLS_DIR", self.tools)
        self.frontend_patch = mock.patch.object(
            installer,
            "FRONTEND_ROOT",
            self.root,
        )
        self.nodes_patch = mock.patch.object(
            installer,
            "CUSTOM_NODES_DIR",
            self.custom_nodes,
        )
        self.tools_patch.start()
        self.frontend_patch.start()
        self.nodes_patch.start()

    def tearDown(self) -> None:
        self.nodes_patch.stop()
        self.frontend_patch.stop()
        self.tools_patch.stop()
        self.temp.cleanup()

    def create_ready_install(self) -> None:
        frontend = self.manifest["frontend"]
        frontend_dir = self.root / frontend["folder"]
        frontend_dir.mkdir(parents=True)
        (frontend_dir / "package.json").write_text(
            json.dumps({"version": frontend["version"]}),
            encoding="utf-8",
        )
        (frontend_dir / "pnpm-lock.yaml").write_text("", encoding="utf-8")
        (frontend_dir / "dist").mkdir(parents=True)
        (frontend_dir / "dist" / "index.html").write_text(
            "<!doctype html>",
            encoding="utf-8",
        )
        installer.frontend_build_marker_path(frontend_dir).write_text(
            json.dumps(
                {
                    "head": frontend["ref"],
                    "ref": frontend["ref"],
                    "version": frontend["version"],
                }
            ),
            encoding="utf-8",
        )

        bridge = self.manifest["nodes"][0]
        bridge_dir = self.custom_nodes / bridge["folder"]
        (bridge_dir / "editor_bridge").mkdir(parents=True)
        (bridge_dir / "__init__.py").write_text("", encoding="utf-8")
        (bridge_dir / "editor_bridge" / "routes.py").write_text(
            "",
            encoding="utf-8",
        )

    def test_preflight_accepts_matching_pinned_install(self) -> None:
        self.create_ready_install()

        def fake_head(path: Path) -> str | None:
            if path == self.root / "ComfyUI_frontend":
                return "f" * 40
            if path == self.custom_nodes / "comfyui-editor-bridge":
                return "b" * 40
            return None

        with mock.patch.object(installer, "repo_head", side_effect=fake_head):
            self.assertEqual(
                installer.editor_integration_errors(self.manifest),
                [],
            )

    def test_preflight_rejects_stale_frontend_build(self) -> None:
        self.create_ready_install()
        marker_path = (
            installer.frontend_build_marker_path(
                self.root / "ComfyUI_frontend"
            )
        )
        marker_path.write_text(
            json.dumps(
                {
                    "head": "0" * 40,
                    "ref": "f" * 40,
                    "version": "1.49.0",
                }
            ),
            encoding="utf-8",
        )

        def fake_head(path: Path) -> str | None:
            if path == self.root / "ComfyUI_frontend":
                return "f" * 40
            if path == self.custom_nodes / "comfyui-editor-bridge":
                return "b" * 40
            return None

        with mock.patch.object(installer, "repo_head", side_effect=fake_head):
            errors = installer.editor_integration_errors(self.manifest)
        self.assertIn(
            "frontend build marker does not match the pinned checkout",
            errors,
        )

    def test_frontend_install_clones_dependencies_builds_and_marks_revision(
        self,
    ) -> None:
        frontend = self.manifest["frontend"]
        frontend_dir = self.root / frontend["folder"]

        def fake_clone(_repo: str, target: Path, _ref: str | None) -> None:
            target.mkdir(parents=True)
            (target / "package.json").write_text(
                json.dumps({"version": frontend["version"]}),
                encoding="utf-8",
            )
            (target / "pnpm-lock.yaml").write_text("", encoding="utf-8")

        def fake_run(command: list[str], *, cwd: Path, env=None) -> None:
            del env
            if "install" in command:
                (cwd / "node_modules").mkdir()
                (cwd / ".pnpm-build-store").mkdir()
            if command[-2:] == ["run", "build"]:
                (cwd / "dist").mkdir()
                (cwd / "dist" / "index.html").write_text(
                    "<!doctype html>",
                    encoding="utf-8",
                )

        with (
            mock.patch.object(installer, "clone_repo", side_effect=fake_clone),
            mock.patch.object(installer, "repo_head", return_value=frontend["ref"]),
            mock.patch.object(installer, "pnpm_prefix", return_value=["pnpm"]),
            mock.patch.object(installer, "run", side_effect=fake_run) as run_mock,
        ):
            installer.install_frontend(self.manifest, install_mode="diff")

        self.assertEqual(run_mock.call_count, 2)
        run_mock.assert_any_call(
            [
                "pnpm",
                "install",
                "--frozen-lockfile",
                "--fetch-retries=5",
                "--fetch-timeout=300000",
                "--network-concurrency=8",
                "--store-dir",
                str(frontend_dir / ".pnpm-build-store"),
            ],
            cwd=frontend_dir,
        )
        run_mock.assert_any_call(
            ["pnpm", "run", "build"],
            cwd=frontend_dir,
        )
        marker = json.loads(
            installer.frontend_build_marker_path(frontend_dir).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(marker["head"], frontend["ref"])
        self.assertEqual(marker["ref"], frontend["ref"])
        self.assertEqual(marker["version"], frontend["version"])
        self.assertFalse((frontend_dir / "node_modules").exists())
        self.assertFalse((frontend_dir / ".pnpm-build-store").exists())

    def test_frontend_install_skips_current_build(self) -> None:
        self.create_ready_install()
        frontend = self.manifest["frontend"]
        frontend_dir = self.root / frontend["folder"]
        (frontend_dir / "node_modules").mkdir()
        (frontend_dir / ".pnpm-build-store").mkdir()

        with (
            mock.patch.object(installer, "clone_repo"),
            mock.patch.object(installer, "repo_head", return_value=frontend["ref"]),
            mock.patch.object(installer, "run") as run_mock,
        ):
            installer.install_frontend(self.manifest, install_mode="diff")

        run_mock.assert_not_called()
        self.assertFalse((frontend_dir / "node_modules").exists())
        self.assertFalse((frontend_dir / ".pnpm-build-store").exists())

    def test_frontend_install_preserves_store_after_build_failure(self) -> None:
        frontend = self.manifest["frontend"]
        frontend_dir = self.root / frontend["folder"]

        def fake_clone(_repo: str, target: Path, _ref: str | None) -> None:
            target.mkdir(parents=True)
            (target / "package.json").write_text(
                json.dumps({"version": frontend["version"]}),
                encoding="utf-8",
            )
            (target / "pnpm-lock.yaml").write_text("", encoding="utf-8")

        def fake_run(command: list[str], *, cwd: Path, env=None) -> None:
            del env
            if "install" in command:
                (cwd / "node_modules").mkdir()
                (cwd / ".pnpm-build-store").mkdir()
                return
            raise RuntimeError("simulated frontend build failure")

        with (
            mock.patch.object(installer, "clone_repo", side_effect=fake_clone),
            mock.patch.object(installer, "repo_head", return_value=frontend["ref"]),
            mock.patch.object(installer, "pnpm_prefix", return_value=["pnpm"]),
            mock.patch.object(installer, "run", side_effect=fake_run),
            self.assertRaisesRegex(RuntimeError, "simulated frontend build failure"),
        ):
            installer.install_frontend(self.manifest, install_mode="diff")

        self.assertFalse((frontend_dir / "node_modules").exists())
        self.assertTrue((frontend_dir / ".pnpm-build-store").exists())

    def test_frontend_node_engine_rejects_wrong_major(self) -> None:
        frontend = {"node_engine": ">=25 <26"}
        completed = mock.Mock(returncode=0, stdout="v24.12.0\n")
        with (
            mock.patch.object(
                installer, "command_prefix", return_value=["node"]
            ),
            mock.patch.object(
                installer.subprocess, "run", return_value=completed
            ),
            self.assertRaisesRegex(SystemExit, r"requires Node.js >=25 <26"),
        ):
            installer.validate_frontend_node_engine(frontend)

    def test_pnpm_prefix_prefers_exact_direct_version_over_corepack(self) -> None:
        completed = mock.Mock(returncode=0, stdout="11.13.1\n")

        def command(name: str) -> list[str]:
            return [name]

        with (
            mock.patch.object(installer, "command_prefix", side_effect=command),
            mock.patch.object(installer.subprocess, "run", return_value=completed),
        ):
            prefix = installer.pnpm_prefix("pnpm@11.13.1")

        self.assertEqual(prefix, ["pnpm"])

    def test_managed_git_command_scopes_safe_directory_to_checkout(self) -> None:
        checkout = self.root / "managed checkout"

        command = installer.managed_git_command(checkout, "rev-parse", "HEAD")

        self.assertEqual(command[0:2], ["git", "-c"])
        self.assertEqual(command[2], f"safe.directory={checkout.resolve().as_posix()}")
        self.assertEqual(command[3:], ["-C", str(checkout.resolve()), "rev-parse", "HEAD"])

    def test_frontend_node_engine_accepts_matching_major(self) -> None:
        frontend = {"node_engine": ">=25 <26"}
        completed = mock.Mock(returncode=0, stdout="v25.9.0\n")
        with (
            mock.patch.object(
                installer, "command_prefix", return_value=["node"]
            ),
            mock.patch.object(
                installer.subprocess, "run", return_value=completed
            ),
        ):
            installer.validate_frontend_node_engine(frontend)

    def test_frontend_migrates_from_legacy_tools_location(self) -> None:
        frontend = self.manifest["frontend"]
        legacy = self.tools / frontend["folder"]
        legacy.mkdir(parents=True)
        (legacy / "preserved.txt").write_text("existing clone", encoding="utf-8")

        installer.migrate_legacy_frontend_checkout(self.manifest)

        target = self.root / frontend["folder"]
        self.assertFalse(legacy.exists())
        self.assertEqual(
            (target / "preserved.txt").read_text(encoding="utf-8"),
            "existing clone",
        )

    def test_bridge_install_replaces_only_legacy_external_link(self) -> None:
        bridge = self.manifest["nodes"][0]
        external = self.root / "legacy-bridge"
        external.mkdir()
        (external / "keep.txt").write_text("preserved", encoding="utf-8")
        target = self.custom_nodes / bridge["folder"]
        target.symlink_to(external, target_is_directory=True)

        def fake_clone(repo: str, clone_target: Path, ref: str | None) -> None:
            self.assertEqual(repo, bridge["repo"])
            self.assertEqual(ref, bridge["ref"])
            clone_target.mkdir()
            (clone_target / "editor_bridge").mkdir()
            (clone_target / "__init__.py").write_text("", encoding="utf-8")
            (clone_target / "editor_bridge" / "routes.py").write_text(
                "",
                encoding="utf-8",
            )

        with mock.patch.object(installer, "clone_repo", side_effect=fake_clone):
            installer.install_git_clone_node(
                bridge,
                "python",
                install_mode="diff",
                no_deps=False,
            )

        self.assertFalse(target.is_symlink())
        self.assertTrue((target / "__init__.py").is_file())
        self.assertEqual(
            (external / "keep.txt").read_text(encoding="utf-8"),
            "preserved",
        )


class InstallStateTests(unittest.TestCase):
    def test_embedded_node_tracks_dependency_files_not_parent_git_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            node = Path(temp) / "embedded-node"
            node.mkdir()
            requirements = node / "requirements.txt"
            requirements.write_text("example-package==1\n", encoding="utf-8")

            with mock.patch.object(installer, "repo_head") as repo_head:
                before = installer.dependency_state(node)
            repo_head.assert_not_called()
            self.assertTrue(before.startswith("deps:"))

            requirements.write_text("example-package==2\n", encoding="utf-8")
            after = installer.dependency_state(node)
            self.assertNotEqual(before, after)

    def test_real_nested_repo_tracks_its_own_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            node = Path(temp) / "nested-repo"
            (node / ".git").mkdir(parents=True)
            with mock.patch.object(installer, "repo_head", return_value="a" * 40):
                self.assertEqual(installer.dependency_state(node), "git:" + "a" * 40)

    def test_legacy_parent_sha_migrates_without_false_dependency_refresh(self) -> None:
        self.assertFalse(
            installer.install_state_entry_changed("a" * 40, "deps:" + "b" * 64)
        )
        self.assertFalse(
            installer.install_state_entry_changed("a" * 40, "git:" + "a" * 40)
        )
        self.assertTrue(
            installer.install_state_entry_changed(
                "git:" + "a" * 40,
                "git:" + "b" * 40,
            )
        )


if __name__ == "__main__":
    unittest.main()
