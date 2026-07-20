"""Local Image Studio: a tiny web page + backend for uncensored image generation.

Owns a single sd-server subprocess and swaps the loaded model on demand (only one
fits the 8 GB GPU at a time). Serves a browser UI at http://127.0.0.1:8090 with a
model dropdown, prompt fields, and inline image display. Stdlib only.
"""
import base64
import json
import os
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "sd-cpp-src", "build", "bin", "sd-server.exe")
SD_PORT = 8083
UI_PORT = 8090

def m(p):
    return os.path.join(ROOT, "models", p)

# name -> (sd-server args, needs score tags?, default steps, default size)
MODELS = {
    "Pony V6 XL (art)": {
        "args": ["-m", m("ponyDiffusionV6XL.safetensors"), "--vae", m("sdxl_vae.safetensors"), "--diffusion-fa"],
        "tags": "score_9, score_8_up, ", "steps": 24, "size": 1024,
    },
    "Juggernaut XL (photo)": {
        "args": ["-m", m("Juggernaut-XL-v9.safetensors"), "--diffusion-fa"],
        "tags": "", "steps": 25, "size": 1024,
    },
    "Chroma1-HD (quality)": {
        "args": ["--diffusion-model", m("Chroma1-HD-Q5_K_S.gguf"), "--t5xxl", m("t5xxl-Q5_K_M.gguf"),
                 "--vae", m("flux-ae.safetensors"), "--backend", "te=cpu", "--diffusion-fa"],
        "tags": "", "steps": 20, "size": 768,
    },
}

_state = {"model": None, "proc": None}

def _stop():
    p = _state["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    _state["proc"] = None
    _state["model"] = None

def ensure_model(name):
    if _state["model"] == name and _state["proc"] and _state["proc"].poll() is None:
        return
    _stop()
    args = [SD] + MODELS[name]["args"] + ["--listen-ip", "127.0.0.1", "--listen-port", str(SD_PORT)]
    _state["proc"] = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # sd-server has no /health and loads the model lazily on first request; readiness
    # just means the HTTP server is listening, which /v1/models reports with a 200.
    deadline = time.time() + 120
    while time.time() < deadline:
        if _state["proc"].poll() is not None:
            raise RuntimeError("sd-server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SD_PORT}/v1/models", timeout=3) as r:
                if r.status == 200:
                    _state["model"] = name
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("sd-server did not start listening in time")

def generate(payload):
    name = payload["model"]
    cfg = MODELS[name]
    ensure_model(name)
    prompt = cfg["tags"] + payload.get("prompt", "").strip()
    body = json.dumps({
        "prompt": prompt,
        "negative_prompt": payload.get("negative", "blurry, low quality, deformed"),
        "width": int(payload.get("size", cfg["size"])),
        "height": int(payload.get("size", cfg["size"])),
        "steps": int(payload.get("steps", cfg["steps"])),
        "cfg_scale": float(payload.get("cfg", 6)),
        "sample_method": payload.get("sampler", "euler_a"),
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{SD_PORT}/v1/images/generations",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return resp["data"][0]["b64_json"], prompt


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._send(200, json.dumps({"models": list(MODELS.keys()),
                                         "current": _state["model"]}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if payload.get("model") not in MODELS:
                raise ValueError("unknown model")
            if not payload.get("prompt", "").strip():
                raise ValueError("empty prompt")
            b64, used = generate(payload)
            self._send(200, json.dumps({"image": b64, "used_prompt": used, "model": payload["model"]}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Local Image Studio</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{color-scheme:dark}
 body{background:#0d0f14;color:#e6e8ec;font:15px/1.5 system-ui,sans-serif;margin:0}
 .wrap{max-width:1100px;margin:0 auto;padding:24px;display:grid;grid-template-columns:340px 1fr;gap:24px}
 h1{grid-column:1/3;margin:0 0 4px;font-size:20px}
 .sub{grid-column:1/3;color:#8b93a1;margin:-4px 0 8px}
 label{display:block;font-size:12px;color:#9aa3b2;margin:12px 0 4px;text-transform:uppercase;letter-spacing:.04em}
 select,textarea,input{width:100%;box-sizing:border-box;background:#161a22;color:#e6e8ec;border:1px solid #2a303c;border-radius:8px;padding:9px 10px;font:inherit}
 textarea{resize:vertical;min-height:70px}
 .row{display:flex;gap:10px}.row>*{flex:1}
 button{margin-top:16px;width:100%;background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:12px;font-weight:600;font-size:15px;cursor:pointer}
 button:disabled{background:#2a303c;color:#6b7280;cursor:default}
 .stage{background:#0a0c11;border:1px solid #1e232c;border-radius:12px;min-height:520px;display:flex;align-items:center;justify-content:center;overflow:hidden}
 .stage img{max-width:100%;max-height:78vh;display:block}
 .status{grid-column:1/3;color:#8b93a1;min-height:20px}
 .hint{color:#6b7280;font-size:12px;margin-top:6px}
</style></head><body><div class=wrap>
 <h1>Local Image Studio</h1>
 <div class=sub>Uncensored, on your own GPU. Switching models reloads (~20s); same model stays warm.</div>
 <div>
  <label>Model</label><select id=model></select>
  <label>Prompt</label><textarea id=prompt placeholder="describe the image..."></textarea>
  <label>Negative prompt</label><input id=neg value="blurry, low quality, deformed">
  <div class=row><div><label>Size</label><select id=size><option>512</option><option selected>768</option><option>1024</option></select></div>
   <div><label>Steps</label><input id=steps type=number value=22 min=4 max=60></div></div>
  <button id=go>Generate</button>
  <div class=hint id=hint></div>
 </div>
 <div class=stage id=stage><span style="color:#4b5563">your image appears here</span></div>
 <div class=status id=status></div>
</div><script>
 const $=id=>document.getElementById(id);
 async function loadModels(){const r=await fetch('/api/models');const d=await r.json();
  $('model').innerHTML=d.models.map(m=>`<option>${m}</option>`).join('');
  const tips={'Pony V6 XL (art)':'Art/anime. score_ tags added for you.','Juggernaut XL (photo)':'Photorealistic. Plain descriptive prompts.','Chroma1-HD (quality)':'Highest quality, follows complex prompts. Slower (~3 min).'};
  $('model').onchange=()=>$('hint').textContent=tips[$('model').value]||'';$('model').onchange();}
 $('go').onclick=async()=>{
  const body={model:$('model').value,prompt:$('prompt').value,negative:$('neg').value,size:+$('size').value,steps:+$('steps').value};
  if(!body.prompt.trim()){$('status').textContent='Enter a prompt first.';return;}
  $('go').disabled=true;const t0=Date.now();
  $('status').textContent='Generating... (first image on a model loads it, ~20-40s extra)';
  try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
   const d=await r.json();
   if(d.error){$('status').textContent='Error: '+d.error;}
   else{$('stage').innerHTML='<img src="data:image/png;base64,'+d.image+'">';
    $('status').textContent='Done in '+((Date.now()-t0)/1000).toFixed(0)+'s. Prompt: '+d.used_prompt;}
  }catch(e){$('status').textContent='Error: '+e;}
  $('go').disabled=false;};
 loadModels();
</script></body></html>"""

if __name__ == "__main__":
    if not os.path.exists(SD):
        raise SystemExit(f"sd-server not found at {SD}")
    print(f"Image Studio: open http://127.0.0.1:{UI_PORT} in your browser")
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()
