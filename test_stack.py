"""
test_stack.py — End-to-end smoke test for the NeuroRecovery stack.

Tests every layer without needing a real MRI file or running frontend.
Run AFTER starting run_all.py (or the two FastAPI services manually).

Usage:
    python test_stack.py
"""

import io, json, struct, gzip, sys, time
import urllib.request
import urllib.error

BASE_VISION = "http://localhost:8001"
BASE_AGENT  = "http://localhost:8000"

OK   = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"

def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

def post_json(url, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
           headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def post_file(url, filename, file_bytes, content_type="application/gzip"):
    boundary = b"----TestBoundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' +
        filename.encode() + b'"\r\n'
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" +
        file_bytes + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(url, data=body, method="POST",
          headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def make_minimal_nifti_gz() -> bytes:
    """
    Build the smallest valid NIfTI-1 .nii.gz in memory (8x8x8, float32).
    No file on disk needed.
    """
    # NIfTI-1 header is exactly 348 bytes
    hdr = bytearray(348)
    # sizeof_hdr = 348
    struct.pack_into("<i", hdr, 0, 348)
    # dim[0]=3, dim[1..3]=8
    struct.pack_into("<8h", hdr, 40, 3, 8, 8, 8, 1, 1, 1, 1)
    # datatype = 16 (float32), bitpix = 32
    struct.pack_into("<h", hdr, 70, 16)
    struct.pack_into("<h", hdr, 72, 32)
    # pixdim (voxel size) = 1mm isotropic
    struct.pack_into("<8f", hdr, 76, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
    # vox_offset = 352 (header + 4-byte extension block)
    struct.pack_into("<f", hdr, 108, 352.0)
    # magic = "n+1\0"
    hdr[344:348] = b"n+1\x00"

    ext   = b"\x00\x00\x00\x00"              # no extensions
    voxels = bytes(8 * 8 * 8 * 4)            # 512 zero float32 voxels

    raw = bytes(hdr) + ext + voxels
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw)
    return buf.getvalue()


def check(label, cond, detail=""):
    status = OK if cond else FAIL
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
    return cond


passed = 0; total = 0

print("\n══════ 1. Vision Service ══════")

try:
    r = get(f"{BASE_VISION}/health")
    total+=1; passed += check("Health probe", r.get("status")=="ok")
except Exception as e:
    print(f"  {FAIL}  Vision service unreachable — is it running on 8001? ({e})")
    sys.exit(1)

r = get(f"{BASE_VISION}/model-info")
total+=1; passed += check("Model info endpoint", "feature_dim" in r, str(r.get("feature_dim")))
total+=1; passed += check("Feature dim = 768", r.get("feature_dim")==768)

nii_gz = make_minimal_nifti_gz()
r = post_file(f"{BASE_VISION}/extract-features", "test_scan.nii.gz", nii_gz)
total+=1; passed += check("Feature extraction accepted", "features" in r)
total+=1; passed += check("Feature vector length = 768", len(r.get("features",[]))==768)
total+=1; passed += check("Volume hash present", bool(r.get("volume_hash")))
feats = r.get("features", [])
print(f"       Sample values: {feats[:4]}")

print("\n══════ 2. Agent API ══════")

try:
    r = get(f"{BASE_AGENT}/health")
    total+=1; passed += check("Health probe", r.get("status")=="ok")
except Exception as e:
    print(f"  {FAIL}  Agent API unreachable — is it running on 8000? ({e})")
    sys.exit(1)

r = get(f"{BASE_AGENT}/graph-schema")
total+=1; passed += check("Graph schema endpoint", "nodes" in r)
total+=1; passed += check("All 3 nodes present",
    {"extract_features_tool","retrieve_literature_node","synthesize_report_node"}
    .issubset(set(r.get("nodes",[]))))

payload = {
    "clinical_metadata": {
        "patient_id": "TEST-001",
        "tumor_grade": "GBM",
        "resection_extent": "gross_total",
        "treatment_protocol": "Stupp",
        "weeks_post_surgery": 12,
        "kps_score": 70,
    },
    "image_features": feats,    # pass real features extracted above
    "session_id": "smoke-test-001",
}
print("  Calling /analyze-recovery (may take 20-120s depending on Ollama)...")
t0 = time.time()
try:
    r = post_json(f"{BASE_AGENT}/analyze-recovery", payload)
    elapsed = round(time.time()-t0, 1)
    total+=1; passed += check("Analysis endpoint returned 200", "final_report" in r, f"{elapsed}s")
    total+=1; passed += check("Report is non-empty", len(r.get("final_report","")) > 50)
    total+=1; passed += check("Session ID echoed back", r.get("session_id")=="smoke-test-001")
    total+=1; passed += check("Warnings field present", "warnings" in r)
    print(f"\n  Report preview (first 300 chars):\n  {r['final_report'][:300]!r}")
except Exception as e:
    total+=1; passed += check("Analysis endpoint", False, str(e))

print(f"\n══════ Result: {passed}/{total} checks passed ══════\n")
sys.exit(0 if passed == total else 1)
