#!/usr/bin/env bash
# Smoke-test a built Docker image before it is published.
#
# Usage:  test/smoke_image.sh <image> [expected-version]
#
# Run from the repository root; the image is fed the datasets under test/ and
# checked against the reference matrices under validation/.
set -euo pipefail

IMAGE="${1:?usage: $0 <image> [expected-version]}"
EXPECTED_VERSION="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }

# Command substitution is used throughout rather than piping into grep: `grep -q`
# exits on its first match, which sends SIGPIPE to docker and trips pipefail.
run_image() {
    docker run --rm -v "$REPO_ROOT":/data:ro "$IMAGE" "$@"
}

echo "== reported version =="
version="$(docker run --rm "$IMAGE" --version)"
echo "$version"
if [ -n "$EXPECTED_VERSION" ] && [ "$version" != "$EXPECTED_VERSION" ]; then
    fail "version is '$version', expected '$EXPECTED_VERSION'"
fi

echo "== output is byte-identical to the C reference =="
run_image -i /data/test/boring.tab -c -s | diff - validation/boring_c.tab \
    || fail "output differs from validation/boring_c.tab"

echo "== runs under an arbitrary UID =="
# Containers are routinely started with `-u $(id -u)`. Numba's on-disk cache
# needs a writable directory and has broken this before, so it is explicit.
docker run --rm -u 12345:12345 -v "$REPO_ROOT":/data:ro "$IMAGE" \
    -i /data/test/boring.tab -c -s | diff - validation/boring_c.tab \
    || fail "output differs when running as an arbitrary UID"

echo "== every missing-data handler runs =="
for y in 0 1 2 3; do
    run_image -i /data/test/boring.tab -c -s -y "$y" > /dev/null \
        || fail "handler -y $y failed"
done

echo "== the bundled pyarrow loader is the one in use =="
arrow_log="$(run_image -i /data/test/boring.tab -c 2>&1 || true)"
case "$arrow_log" in
    *"Arrow fast path"*) : ;;
    *) fail "the Arrow loader was not used; is pyarrow present in the image?" ;;
esac

echo "== triangular output formats =="
for fmt in full lower-tri upper-tri; do
    rows="$(run_image -i /data/test/boring.tab -c -s -m "$fmt" | wc -l)"
    [ "$rows" -eq 6 ] || fail "format $fmt produced $rows lines, expected 6"
done

echo "All image smoke tests passed"
