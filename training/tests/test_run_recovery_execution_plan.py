import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from run_recovery_execution_plan import (
    latest_checkpoint,
    load_state,
    new_state,
    read_plan,
    run,
    validate_stages,
)


class RecoveryExecutionPlanRunnerTests(unittest.TestCase):
    def stage(self, name, requires, gpu=1, physical="1"):
        environment = {"CUDA_VISIBLE_DEVICES": physical} if gpu else {}
        return {
            "name": name,
            "requires": requires,
            "environment": environment,
            "gpu_count": gpu,
            "argv": ["/python", "/stage.py", "--device", "cuda:0"],
            "outputs": [f"/{name}.json"],
        }

    def test_stages_require_serial_dependencies_and_non_greppy_gpu(self):
        document = {
            "stages": [
                self.stage("train_recovery", ["admission"]),
                self.stage("pack", ["train_recovery:status=complete"]),
                self.stage("compare", ["pack"], gpu=0),
            ]
        }
        self.assertEqual(len(validate_stages(document)), 3)
        document["stages"][0]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(ValueError, "non-Greppy"):
            validate_stages(document)

    def test_stages_reject_forward_dependency_and_extra_environment(self):
        with self.assertRaisesRegex(ValueError, "unsatisfied dependency"):
            validate_stages({"stages": [self.stage("pack", ["train_recovery"])]})
        stage = self.stage("train", ["admission"])
        stage["environment"]["HF_HOME"] = "/tmp/cache"
        with self.assertRaisesRegex(ValueError, "unadmitted environment"):
            validate_stages({"stages": [stage]})

    def test_latest_checkpoint_uses_highest_complete_numbered_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "recovery-step-000025.safetensors").write_bytes(b"a")
            (root / "recovery-step-000100.safetensors").write_bytes(b"b")
            (root / "recovery-final-step-000125.safetensors").write_bytes(b"c")
            argv = ["python", "train.py", "--checkpoint-dir", str(root)]
            self.assertEqual(
                latest_checkpoint(argv), root / "recovery-step-000100.safetensors"
            )

    def test_plan_and_state_are_bound_to_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            document = {
                "format": "ctox.recovery.execution-plan.v1",
                "status": "admitted",
                "execution_order": "serial",
            }
            plan.write_text(json.dumps(document), encoding="utf-8")
            loaded, digest = read_plan(plan)
            self.assertEqual(loaded, document)
            self.assertEqual(digest, hashlib.sha256(plan.read_bytes()).hexdigest())
            state = new_state(plan, digest)
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(load_state(state_path, plan, digest)["plan_sha256"], digest)
            with self.assertRaisesRegex(ValueError, "exact plan"):
                load_state(state_path, plan, "0" * 64)

    def test_runner_executes_stage_and_hashes_output_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "stage.py"
            output = root / "output.json"
            script.write_text(
                "import os, pathlib, sys\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'\n"
                "pathlib.Path(sys.argv[1]).write_text('passed\\n')\n",
                encoding="utf-8",
            )
            python = Path(sys.executable).resolve()
            plan = root / "plan.json"
            document = {
                "format": "ctox.recovery.execution-plan.v1",
                "status": "admitted",
                "execution_order": "serial",
                "implementation": {
                    "python": str(python),
                    "scripts": {
                        "stage.py": {
                            "path": str(script),
                            "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                        }
                    },
                },
                "stages": [
                    {
                        "name": "train_recovery",
                        "requires": ["admission"],
                        "environment": {"CUDA_VISIBLE_DEVICES": "1"},
                        "gpu_count": 1,
                        "argv": [
                            str(python),
                            str(script),
                            str(output),
                            "--device",
                            "cuda:0",
                        ],
                        "outputs": [str(output)],
                    }
                ],
            }
            plan.write_text(json.dumps(document), encoding="utf-8")
            state = root / "state.json"
            run(plan, state, resume=False, dry_run=False)
            persisted = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "complete")
            self.assertEqual(
                persisted["completed"][0]["outputs"][0]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
