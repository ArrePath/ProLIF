# Operator-owned internal publication

Ariadne keeps consuming an immutable GitHub archive. Publishing this distribution
internally does not change that source override. Do not publish to public PyPI,
overwrite an occupied version, enable inherited release workflows or merge/deploy
as part of this procedure.

## Build the reviewed source

Check out the exact reviewed commit recorded in the fork/Ariadne PRs in an isolated
worktree. Verify a clean source tree and `version = "2.2.3+arrepath.1"` before building.
Publish only after the PR records the qualified artifact digests and the build's
`SOURCE_DATE_EPOCH`.

```bash
set -euo pipefail
version='2.2.3+arrepath.1'
test -z "$(git status --porcelain)"
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
uv build --sdist --wheel --out-dir dist
uvx twine check "dist/prolif-${version}.tar.gz" "dist/prolif-${version}-py3-none-any.whl"
sha256sum "dist/prolif-${version}.tar.gz" "dist/prolif-${version}-py3-none-any.whl"
```

Use a fresh output directory; upload only these two qualified files, not a wildcard
containing unrelated releases. Require separate built-wheel and released-consumer
qualification before publication. Do not treat successful building alone as
scientific qualification of a different source commit.

## Publish with your authorized AWS identity

The workstation instance role is read-only for CodeArtifact. Run locally with an
existing authorized publisher profile; do not broaden IAM or copy credentials to
the workstation. Set `PUBLISH_PROFILE` yourself. Never use shell tracing around
token retrieval or upload, and do not persist the token in a configuration file.

```bash
set -euo pipefail
set +x
: "${PUBLISH_PROFILE:?Set your authorized publisher profile}"
aws --profile "$PUBLISH_PROFILE" sts get-caller-identity
occupied="$(aws --profile "$PUBLISH_PROFILE" --region us-east-1 codeartifact list-package-versions \
  --domain arrepath --domain-owner 104165404940 --repository arrepath \
  --format pypi --package prolif \
  --query "versions[?version=='${version}'].version" --output text)"
if [ -n "$occupied" ]; then
  printf "Refusing to publish occupied version: %s\n" "$version" >&2
  exit 1
fi
```

Stop if `2.2.3+arrepath.1` already exists. Do not delete/re-upload it or use
`--skip-existing` as proof of a successful release. Resolve an occupied version
with the owner before building/publishing a new version.

Run with a single publisher; the preflight is not an atomic version reservation.
After confirming the version is unoccupied:

```bash
export UV_PUBLISH_USERNAME=aws
export UV_PUBLISH_PASSWORD="$(aws --profile "$PUBLISH_PROFILE" --region us-east-1 \
  codeartifact get-authorization-token --domain arrepath --domain-owner 104165404940 \
  --duration-seconds 900 --query authorizationToken --output text)"
uv publish \
  --publish-url 'https://arrepath-104165404940.d.codeartifact.us-east-1.amazonaws.com/pypi/arrepath/' \
  "dist/prolif-${version}.tar.gz" "dist/prolif-${version}-py3-none-any.whl"
unset UV_PUBLISH_PASSWORD UV_PUBLISH_USERNAME
```

## Verify remote bytes and installation

```bash
aws --profile "$PUBLISH_PROFILE" --region us-east-1 codeartifact list-package-version-assets \
  --domain arrepath --domain-owner 104165404940 --repository arrepath \
  --format pypi --package prolif --package-version "$version"
mkdir -p verified-dist
for asset in "prolif-${version}.tar.gz" "prolif-${version}-py3-none-any.whl"; do
  aws --profile "$PUBLISH_PROFILE" --region us-east-1 codeartifact get-package-version-asset \
    --domain arrepath --domain-owner 104165404940 --repository arrepath \
    --format pypi --package prolif --package-version "$version" \
    --asset "$asset" "verified-dist/$asset"
  cmp "dist/$asset" "verified-dist/$asset"
done
sha256sum verified-dist/*
```

Use an isolated verification environment outside the source checkout, with the
normal CodeArtifact keyring plugin and your authorized profile. Install the exact
published wheel/version, not an editable checkout; verify `importlib.metadata.version("prolif")`
and `prolif.__file__`. Record repository, version, reviewed source commit, remote
asset hashes and install checks in the release/PR evidence. Publication and remote
readback have not been performed by the agent.
