import json, sys, os

chat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat.json')
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_real_extracted.txt')

print(f"Loading {chat_path}...")
with open(chat_path, encoding='utf-8') as f:
    data = json.load(f)

print(f"JSON keys: {list(data.keys())[:10]}")

requests = data.get('requests', [])
print(f"Total requests: {len(requests)}")

# Inspect structure of first request
if requests:
    r = requests[0]
    print(f"First request keys: {list(r.keys())}")
    turn = r.get('turn', {})
    if turn:
        print(f"Turn keys: {list(turn.keys())}")
        req = turn.get('request', {})
        if req:
            print(f"Request keys: {list(req.keys())}")
            msg = req.get('message', {})
            if msg:
                print(f"Message keys: {list(msg.keys())}")
                print(f"Text preview: {str(msg.get('text',''))[:200]}")

# Try different structures
lines = []
for i, r in enumerate(requests):
    # Try various possible paths
    candidates = []
    
    # Path 1: r -> turn -> request -> message -> text
    try: candidates.append(('USER_v1', r['turn']['request']['message']['text']))
    except: pass
    
    # Path 2: r -> turn -> response -> value (array of markdown)
    try:
        resp_parts = r['turn']['response']['value']
        if isinstance(resp_parts, list):
            full = ' '.join(p.get('value','') for p in resp_parts if isinstance(p,dict))
            if full.strip():
                candidates.append(('COPILOT_v1', full))
        elif isinstance(resp_parts, str):
            candidates.append(('COPILOT_v1', resp_parts))
    except: pass
    
    # Path 3: r -> message
    try: candidates.append(('MSG', r['message']['text']))
    except: pass
    
    for label, text in candidates:
        if text and len(str(text).strip()) > 5:
            lines.append(f'\n=== REQUEST {i+1} | {label} ===')
            lines.append(str(text).strip()[:2000])

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Written {len(lines)} lines to {out_path}")
