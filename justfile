# ── Cargo targets ────────────────────────────────────────────────

build:
    cargo build

test:
    cargo test

clippy:
    cargo clippy -- -D warnings

fmt:
    cargo fmt

check:
    cargo fmt --check
    cargo clippy -- -D warnings

release:
    cargo build --release

clean:
    cargo clean

# ── PMAT targets ─────────────────────────────────────────────────

context:
    pmat context --output context.md

tdg:
    pmat analyze tdg

score:
    pmat rust-project-score

quality: check test tdg

pmat-check:
    pmat analyze tdg
    pmat rust-project-score

pmat-baseline:
    mkdir -p .pmat
    pmat tdg baseline create --output .pmat/baseline.json
