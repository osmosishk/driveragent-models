"""End-to-end tests for da-models with a local file:// bucket.

Run on a Jetson (needs trtexec, tensorrt, onnx, onnxruntime):
    python3 -m unittest discover -s tests -v
"""

import importlib.machinery
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "da-models"


def load_tool():
    loader = importlib.machinery.SourceFileLoader("da_models", str(TOOL))
    spec = importlib.util.spec_from_loader("da_models", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def make_onnx(path):
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def sh(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="da-models-test-"))
        self.bucket = self.tmp / "bucket"
        self.bucket.mkdir()
        self.root = self.tmp / "store"
        self.root.mkdir()
        self.manifest = self.tmp / "models.yaml"
        self.manifest.write_text("schema_version: 1\nbucket: gs://unused\nmodels: {}\n")
        make_onnx(self.tmp / "tiny.onnx")
        # Code repo with a pushed commit.
        self.origin = self.tmp / "origin.git"
        sh(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)])
        self.code = self.tmp / "code"
        sh(["git", "init", "-q", "-b", "main", str(self.code)])
        for k, v in (("user.name", "test"), ("user.email", "test@example.com")):
            sh(["git", "config", k, v], cwd=self.code)
        (self.code / "runtime" / "tiny").mkdir(parents=True)
        (self.code / "runtime" / "tiny" / "run.py").write_text("print('tiny')\n")
        (self.code / "other.txt").write_text("not in the archive\n")
        self.commit_push("init")
        self.env = dict(os.environ, DA_MODELS_MANIFEST=str(self.manifest), DA_MODELS_ROOT=str(self.root),
                        DA_MODELS_BUCKET=f"file://{self.bucket}")

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            if p.is_dir() and not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def commit_push(self, msg):
        sh(["git", "add", "-A"], cwd=self.code)
        sh(["git", "commit", "-q", "-m", msg], cwd=self.code)
        sh(["git", "push", "-q", str(self.origin), "main"], cwd=self.code)
        sh(["git", "fetch", "-q", str(self.origin), "main:refs/remotes/origin/main"], cwd=self.code)

    def spec(self, version="1.0.0", status="production", model="tiny", pyyaml="'>=5.0'"):
        p = self.tmp / f"{model}-{version}.yaml"
        p.write_text(textwrap.dedent(f"""\
            model: {model}
            version: {version}
            status: {status}
            description: test model
            requires: {{jetpack_min: "6.0", tensorrt_min: "10.0", onnxruntime_min: "1.10",
                        python: {{pyyaml: {pyyaml}}}}}
            code: {{repo: {self.code}, subdir: runtime/tiny}}
            files:
              - {{name: tiny.onnx, path: {self.tmp / 'tiny.onnx'}}}
            components:
              main:
                runtime: tensorrt
                source: tiny.onnx
                precision: fp16
                engine: tiny.engine
                trtexec_flags: "--fp16 --memPoolSize=workspace:256"
              ctrl:
                runtime: onnxruntime-cpu
                source: tiny.onnx
                precision: fp32
            preprocessing: {{note: none}}
            """))
        return p

    def run_tool(self, *argv, env=None, ok=True):
        r = subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True,
                           env=env or self.env)
        out = r.stdout + r.stderr
        if ok and r.returncode != 0:
            self.fail(f"da-models {' '.join(argv)} failed:\n{out}")
        if not ok and r.returncode == 0:
            self.fail(f"da-models {' '.join(argv)} did not fail:\n{out}")
        return out

    def publish(self, *a, **kw):
        return self.run_tool("publish", str(self.spec(*a, **kw)))


class TestPublishPull(Env):
    def test_publish_pull_verify(self):
        self.publish()
        man = yaml.safe_load(self.manifest.read_text())
        e = man["models"]["tiny"]["versions"]["1.0.0"]
        self.assertEqual(e["status"], "production")
        self.assertEqual({f["name"] for f in e["files"]}, {"tiny.onnx", "runtime.tar.gz"})
        self.assertEqual(e["components"]["main"]["inputs"][0]["shape"], [1, 3, 8, 8])
        self.assertEqual(len(e["runtime"]["git_commit"]), 40)
        self.assertTrue((self.bucket / "models/tiny/1.0.0/model.json").is_file())

        self.run_tool("pull", "tiny")
        self.assertEqual(os.readlink(self.root / "tiny" / "current"), "1.0.0")
        self.assertEqual(stat.S_IMODE((self.root / "tiny/1.0.0").stat().st_mode), 0o755)
        self.assertEqual((self.root / "tiny/1.0.0/runtime/run.py").read_text(), "print('tiny')\n")
        self.assertFalse((self.root / "tiny/1.0.0/runtime/other.txt").exists())
        self.assertEqual(list((self.root / ".tmp").iterdir()), [])
        self.assertIn("already", self.run_tool("pull", "tiny"))
        self.assertIn("OK", self.run_tool("verify"))
        self.assertIn("OK", self.run_tool("verify", "--remote"))

    def test_version_is_immutable(self):
        self.publish()
        self.assertIn("immutable", self.run_tool("publish", str(self.spec()), ok=False))
        # Even when models.yaml loses the entry, the bucket blocks a second publish.
        self.manifest.write_text("schema_version: 1\nbucket: gs://unused\nmodels: {}\n")
        self.assertIn("already published", self.run_tool("publish", str(self.spec()), ok=False))

    def test_publish_needs_clean_pushed_tree(self):
        (self.code / "runtime" / "tiny" / "new.py").write_text("x = 1\n")
        self.assertIn("not clean", self.run_tool("publish", str(self.spec()), ok=False))
        sh(["git", "add", "-A"], cwd=self.code)
        sh(["git", "commit", "-q", "-m", "local only"], cwd=self.code)
        self.assertIn("not on a remote branch", self.run_tool("publish", str(self.spec()), ok=False))
        self.assertEqual(list(self.bucket.rglob("*.onnx")), [])

    def test_selection_candidate_archived(self):
        self.publish("1.0.0", "production")
        self.publish("1.1.0", "candidate")
        self.publish("0.1.0", "archived", model="old")
        self.run_tool("pull", "tiny")
        self.assertEqual(os.readlink(self.root / "tiny" / "current"), "1.0.0")
        self.run_tool("pull", "tiny@1.1.0")
        self.assertEqual(os.readlink(self.root / "tiny" / "current"), "1.0.0")
        self.assertIn("no production version", self.run_tool("pull", "old", ok=False))
        self.run_tool("pull", "old@0.1.0")
        self.assertFalse((self.root / "old" / "current").exists())
        out = self.run_tool("list")
        self.assertIn("archived", out)
        self.assertIn("candidate", out)
        self.run_tool("pull", "tiny@1.1.0", "--set-current")
        self.assertEqual(os.readlink(self.root / "tiny" / "current"), "1.1.0")

    def test_pull_rejects_bad_sha256(self):
        self.publish()
        obj = self.bucket / "models/tiny/1.0.0/tiny.onnx"
        data = bytearray(obj.read_bytes())
        data[-1] ^= 0xFF
        obj.write_bytes(bytes(data))
        self.assertIn("sha256", self.run_tool("pull", "tiny", ok=False))
        self.assertFalse((self.root / "tiny" / "1.0.0").exists())
        self.assertEqual(list((self.root / ".tmp").iterdir()), [])

    def test_verify_detects_local_change(self):
        self.publish()
        self.run_tool("pull", "tiny")
        (self.root / "tiny/1.0.0/runtime/run.py").write_text("print('changed')\n")
        self.assertIn("runtime file changed", self.run_tool("verify", ok=False))


class TestBuild(Env):
    def test_build_upload_then_cache_hit(self):
        self.publish()
        self.run_tool("pull", "tiny")
        out = self.run_tool("build", "tiny")
        self.assertIn("uploaded", out)
        engines = list(self.bucket.glob("engines/tiny/1.0.0/*/tiny.engine"))
        self.assertEqual(len(engines), 1)
        tag = engines[0].parent.name
        self.assertRegex(tag, r"^[a-z0-9]+-jp[\d.]+-trt[\d.]+-fp16$")
        meta = json.loads((engines[0].parent / "tiny.engine.json").read_text())
        self.assertEqual(meta["origin"], "trtexec")
        self.assertIn("ok: local engine", self.run_tool("build", "tiny"))

        # Second device, same tag: it must use the cache and never run trtexec.
        root_b = self.tmp / "store_b"
        root_b.mkdir()
        env_b = dict(self.env, DA_MODELS_ROOT=str(root_b), TRTEXEC="/bin/false")
        self.run_tool("pull", "tiny", env=env_b)
        self.assertIn("cache hit", self.run_tool("build", "tiny", env=env_b))
        self.assertIn("OK", self.run_tool("verify", "--deep", env=env_b))

    def test_build_no_upload_then_upload_verified_engine(self):
        self.publish()
        self.run_tool("pull", "tiny")
        self.assertIn("built", self.run_tool("build", "tiny", "--no-upload"))
        self.assertEqual(list(self.bucket.glob("engines/**/*.engine")), [])
        local = next(self.root.glob("tiny/1.0.0/engines/*/tiny.engine"))
        before = local.read_bytes()
        out = self.run_tool("build", "tiny")
        self.assertIn("ok: local engine", out)
        self.assertIn("uploaded", out)
        cached = next(self.bucket.glob("engines/tiny/1.0.0/*/tiny.engine"))
        self.assertEqual(cached.read_bytes(), before)
        self.assertNotIn("uploaded", self.run_tool("build", "tiny"))

    def test_python_package_requirement(self):
        self.run_tool("publish", str(self.spec(pyyaml="'==0.0.1'")))
        man = yaml.safe_load(self.manifest.read_text())
        self.assertIn("pyyaml", man["models"]["tiny"]["versions"]["1.0.0"]["tested_with"]["python"])
        self.run_tool("pull", "tiny")
        self.assertIn("needs Python package pyyaml==0.0.1", self.run_tool("build", "tiny", ok=False))

    def test_build_does_not_fail_when_upload_denied(self):
        self.publish()
        root_c = self.tmp / "store_c"
        root_c.mkdir()
        env_c = dict(self.env, DA_MODELS_ROOT=str(root_c), DA_DEVICE="orinnx16")
        self.run_tool("pull", "tiny", env=env_c)
        for p in [self.bucket, *self.bucket.rglob("*")]:
            if p.is_dir():
                p.chmod(0o555)
        out = self.run_tool("build", "tiny", env=env_c)
        self.assertIn("cache miss", out)
        self.assertIn("upload not permitted", out)
        self.assertTrue(list(root_c.glob("tiny/1.0.0/engines/orinnx16-*/tiny.engine")))


class TestUnits(unittest.TestCase):
    def test_version_order_and_compare(self):
        t = load_tool()
        vs = ["1.9.0", "1.10.0", "1.10.0-rc1", "0.1.0"]
        self.assertEqual(sorted(vs, key=t.parse_version), ["0.1.0", "1.9.0", "1.10.0-rc1", "1.10.0"])
        self.assertTrue(t.version_ge("10.3.0", "10.3"))
        self.assertFalse(t.version_ge("6.1", "6.2"))
        with self.assertRaises(t.DAError):
            t.parse_version("1.0")

    def test_engine_tag_differs_by_tensorrt_version(self):
        t = load_tool()
        comp = {"runtime": "tensorrt", "precision": "fp16"}

        def tag(module, jp, trtv):
            d = t.Device()
            d.__dict__.update(module=module, jetpack=jp, trt=trtv)  # fill the cached properties
            return d.tag_for(comp)
        nx = tag("orinnx16", "6.0", "8.6.2")
        agx = tag("agxorin64", "6.2.1", "10.3.0")
        self.assertEqual(nx, "orinnx16-jp6.0-trt8.6.2-fp16")
        self.assertNotEqual(tag("orinnx16", "6.0", "8.6.2"), tag("orinnx16", "6.0", "10.3.0"))
        self.assertNotEqual(nx, agx)
        self.assertEqual(t.L4T_TO_JETPACK["36.3.0"], "6.0")

    def test_apply_workspace(self):
        t = load_tool()
        self.assertEqual(t.apply_workspace(["--fp16", "--memPoolSize=workspace:4096", "--noTF32"], 1024),
                         ["--fp16", "--noTF32", "--memPoolSize=workspace:1024"])
        self.assertEqual(t.apply_workspace(["--workspace=4096"], 512), ["--memPoolSize=workspace:512"])
        self.assertEqual(t.apply_workspace([], 256), ["--memPoolSize=workspace:256"])

    def test_runtime_archive_is_reproducible(self):
        t = load_tool()
        env = Env("setUp")
        env.setUp()
        try:
            a, b = env.tmp / "a.tgz", env.tmp / "b.tgz"
            code = {"repo": str(env.code), "subdir": "runtime/tiny"}
            t.make_runtime_archive(code, a)
            t.make_runtime_archive(code, b)
            self.assertEqual(t.sha256_file(a), t.sha256_file(b))
        finally:
            env.tearDown()


if __name__ == "__main__":
    unittest.main()
