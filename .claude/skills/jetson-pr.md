# Create a jetson-containers Pull Request

Help the user create a well-formed PR that follows this repository's conventions.

## Repository PR conventions

- **Target branch**: PRs go to `dev`, not `master`. The CI workflow `pr-on-dev.yml` runs self-hosted Jetson builds against `dev`.
- **Scope**: each PR should touch only the packages it claims to change. Keep unrelated cleanup out.
- **Commit style**: short, lowercase, imperative subject. No period. Examples from git log: `fix`, `cuda 13.2 needs newer drivers to run in Orin`, `tensorrt compatible orin`, `Update ffmpeg version from 8.1 to 8.0.1`.
- **No force-push** to shared branches.

## Step-by-step workflow

1. **Understand the changes**: run `git diff master...HEAD` and `git log master..HEAD --oneline` to see exactly what is changing. Read the changed files if needed to understand the intent.

2. **Identify affected packages**: map every changed file back to its package directory (`packages/<category>/<name>/`). List them — this drives the PR title and body.

3. **Stage and commit** (if there are uncommitted changes):
   - Stage only the files related to this PR.
   - Write a commit message following the style above. No trailing period.
   - Do not skip hooks (`--no-verify`).

4. **Create or switch to a feature branch** if on `master`:
   ```
   git checkout -b <short-descriptive-name>
   ```
   Branch name: lowercase, hyphenated, describes the change (e.g. `upgrade-pytorch-2.9`, `add-sglang-0.4`, `fix-tensorrt-orin`).

5. **Push and open the PR**:
   ```
   gh pr create --base dev --title "<title>" --body "<body>"
   ```

6. **PR title format**: `<verb> <package(s)> <version or brief description>`
   - Version bumps: `upgrade pytorch to 2.9`
   - New packages: `add sglang 0.4`
   - Bug fixes: `fix tensorrt build on Orin`
   - Multi-package: list up to 3, then `and N others`

7. **PR body template**:
   ```
   ## Changes
   - <package>: <what changed and why>

   ## Packages affected
   - `packages/<category>/<name>/`

   ## Test plan
   - [ ] `jetson-containers build <package>` on Orin (JP6)
   - [ ] Add any version-specific test steps here

   ## Notes
   <Any JetPack version constraints, driver requirements, or upstream release links>
   ```

8. **Return the PR URL** when done.

## What good PRs look like in this repo

- Version upgrades include the new version entry in `config.py` **and** update `version.py` if it exists.
- New packages include a `Dockerfile`, metadata (YAML header or `config.yaml`), and at minimum a stub `test.py` or `test.sh`.
- Dependency bumps update `requires` constraints if the new version drops support for an older JetPack.
- PRs do not mix version bumps with unrelated refactors.

## Before opening the PR, verify

- `git diff --stat dev...HEAD` shows only the intended files.
- No `.env`, secrets, or build artifacts are staged.
- The branch name and title are consistent.
