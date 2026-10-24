"""OpenAI Chat Completions gateway between the offline eval containers and a model endpoint.

The eval containers sit on an internal (no-internet) docker network and only see this gateway, so the agent's
shell stays offline and never sees the API key. The gateway adds the key, optionally rewrites the request for the
upstream, and logs every request/response pair (including reasoning) to <log_dir>/<worker>.jsonl.

Backends:
  passthrough  forward as-is (vLLM / SGLang / any server that accepts the harness's extra fields:
               chat_template_kwargs, thinking_token_budget, message.reasoning)
  strict       drop those non-standard fields for strict OpenAI-compatible APIs; history keeps reasoning_content
  openrouter   OpenRouter (https://openrouter.ai/api/v1): thinking on -> reasoning={"enabled": true} (Gemma 4 has
               reasoning off by default there); history reasoning is sent back as the exact `reasoning_details`
               the model returned when the gateway has them (OpenRouter asks for that with tool calls), else as
               the `reasoning` string
--extra-body merges a JSON object into every request, e.g. OpenRouter provider routing:
  '{"provider": {"quantizations": ["bf16"], "allow_fallbacks": false}}'
Route: http://<gateway>:<port>/w/<worker>/v1/chat/completions (or /v1/chat/completions -> worker "default").

usage: python gateway.py --upstream https://host/v1 --log-dir DIR [--backend passthrough] [--model NAME] [--extra-body JSON]
env:   UPSTREAM_API_KEY
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', required=True, help='base URL ending in /v1')
ap.add_argument('--backend', choices=['passthrough', 'strict', 'openrouter'], default='passthrough')
ap.add_argument('--model', default=None, help='model name sent upstream (default: keep the request\'s)')
ap.add_argument('--extra-body', default='', help='JSON object merged into every upstream request')
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--log-dir', required=True)
ap.add_argument('--max-retries', type=int, default=8)
ap.add_argument('--timeout', type=int, default=900, help='per-request upstream timeout (s)')
args = ap.parse_args()
BASE = args.upstream.rstrip('/')
KEY = os.environ.get('UPSTREAM_API_KEY', '')
EXTRA = json.loads(args.extra_body) if args.extra_body else {}
os.makedirs(args.log_dir, exist_ok=True)
_locks, _glock = {}, threading.Lock()
STATS = {'requests': 0, 'errors': 0, 'retries': 0}
_details = {}  # sha256(reasoning text) -> reasoning_details returned with it (openrouter)

NON_STANDARD = ('chat_template_kwargs', 'thinking_token_budget')


def _key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _thinking(req):
    on = (req.get('chat_template_kwargs') or {}).get('enable_thinking')
    return (req.get('thinking_token_budget') or 0) > 0 if on is None else bool(on)


def to_strict(req):
    body = {k: v for k, v in req.items() if k not in NON_STANDARD}
    body['messages'] = [{k: v for k, v in m.items() if k != 'reasoning'} for m in req.get('messages', [])]
    return body


def to_openrouter(req):
    body = {k: v for k, v in req.items() if k not in NON_STANDARD}
    msgs = []
    for m in req.get('messages', []):
        r = m.get('reasoning') or m.get('reasoning_content')
        m = {k: v for k, v in m.items() if k not in ('reasoning', 'reasoning_content')}
        if r and m.get('role') == 'assistant':
            d = _details.get(_key(r))
            if d:
                m['reasoning_details'] = d
            else:
                m['reasoning'] = r
        msgs.append(m)
    body['messages'] = msgs
    if _thinking(req):
        body['reasoning'] = {'enabled': True}
    return body


def call_upstream(body):
    data = json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    if KEY:
        headers['Authorization'] = 'Bearer ' + KEY
    for attempt in range(args.max_retries + 1):
        try:
            req = urllib.request.Request(BASE + '/chat/completions', data, headers)
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                resp = json.load(r)
            if resp.get('error') and not resp.get('choices'):  # some routers report upstream failures with a 200
                err = resp['error'] if isinstance(resp['error'], dict) else {'message': str(resp['error'])}
                code = int(err.get('code') or 502) if str(err.get('code') or '').isdigit() else 502
                if code in (408, 429, 500, 502, 503, 504) and attempt < args.max_retries:
                    STATS['retries'] += 1
                    time.sleep(min(60, 2 ** attempt) * (0.8 + 0.4 * random.random()))
                    continue
                return code, resp
            return 200, resp
        except urllib.error.HTTPError as e:
            payload = e.read().decode(errors='replace')
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < args.max_retries:
                STATS['retries'] += 1
                time.sleep(min(60, 2 ** attempt) * (0.8 + 0.4 * random.random()))
                continue
            return e.code, {'error': {'message': payload[:2000], 'type': 'upstream_error', 'code': e.code}}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < args.max_retries:
                STATS['retries'] += 1
                time.sleep(min(60, 2 ** attempt))
                continue
            return 502, {'error': {'message': str(e), 'type': 'gateway_error'}}


def complete(req):
    if args.model:
        req = {**req, 'model': args.model}
    body = {'strict': to_strict, 'openrouter': to_openrouter}.get(args.backend, dict)(req)
    body.update(EXTRA)
    code, resp = call_upstream(body)
    if code == 200:
        for c in resp.get('choices', []):
            m = c.get('message') or {}
            r = m.get('reasoning_content') or m.get('reasoning')
            if r:  # vLLM reads message.reasoning, ADK sends reasoning_content: return both
                m['reasoning_content'] = m['reasoning'] = r
                if m.get('reasoning_details'):
                    _details[_key(r)] = m['reasoning_details']
    return code, body, resp


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            return self._send(200, {'object': 'list', 'data': [{'id': args.model or 'model', 'object': 'model'}]})
        if self.path.startswith('/stats'):
            return self._send(200, STATS)
        return self._send(200, {'status': 'ok'})

    def do_POST(self):
        m = re.match(r'^(?:/w/([\w.-]+))?/v1/chat/completions$', self.path)
        if not m:
            return self._send(404, {'error': {'message': f'unknown path {self.path}'}})
        worker = m.group(1) or 'default'
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        t0 = time.time()
        code, sent, resp = complete(req)
        STATS['requests'] += 1
        STATS['errors'] += code != 200
        with _glock:
            lock = _locks.setdefault(worker, threading.Lock())
        with lock, open(os.path.join(args.log_dir, f'{worker}.jsonl'), 'a') as f:
            f.write(json.dumps({'ts': t0, 'dt': round(time.time() - t0, 2), 'status': code, 'request': req,
                                'upstream_request': sent, 'response': resp}) + '\n')
        self._send(code, resp)

    def log_message(self, *a):
        pass


print(f'gateway ({args.backend}) on :{args.port} -> {BASE} model={args.model or "<from request>"} '
      f'key={"set" if KEY else "none"} extra_body={sorted(EXTRA)}', flush=True)
ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever()
