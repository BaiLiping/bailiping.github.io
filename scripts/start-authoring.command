#!/bin/zsh
set -eu
cd "${0:A:h}/.."
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
editor_url="http://127.0.0.1:4317/__authoring/"
editor_is_here() {
  curl -fsS "${editor_url}health" 2>/dev/null | node -e 'let s="";process.stdin.on("data",c=>s+=c);process.stdin.on("end",()=>{try{const v=JSON.parse(s);process.exit(v.app==="bailiping-authoring"&&v.root===process.cwd()?0:1)}catch{process.exit(1)}})'
}
if editor_is_here; then
  open "$editor_url"
  exit 0
fi
node scripts/authoring-server.mjs &
editor_pid=$!
trap 'kill "$editor_pid" 2>/dev/null || true' EXIT INT TERM
for attempt in {1..30}; do
  if editor_is_here; then
    open "$editor_url"
    print "The presentation editor is open. Keep this window running while you edit."
    wait "$editor_pid"
    exit $?
  fi
  kill -0 "$editor_pid" 2>/dev/null || { print "Could not start the editor. Check the error above."; exit 1; }
  sleep 0.2
done
print "The editor did not start. Ask Codex to check the local server."
exit 1
