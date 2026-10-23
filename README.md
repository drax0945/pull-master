# pull-master

A harness that takes a GitHub repository link, reads its open issues through the
[GitHub MCP server](https://github.com/github/github-mcp-server), has Claude fix each one in a
local scratch clone, runs the test suite, and merges the fixes that pass into the base branch.

```
GitHub repo link
   │
   ▼
GitHub MCP ── list_issues / issue_read ──► pick issues (trusted authors only, by default)
   │
   ▼
local sandbox: scratch clone + scrubbed shell env (+ virtualenv for Python projects)
   │  baseline test run on main
   ▼
for each issue:
   branch pull-master/issue-N-… ─► Claude (bash, file editor, run_tests, read-only GitHub tools)
        └─► submit_result ─► harness runs the tests ─► fail: output goes back to Claude (up to 3 tries)
                                                    └► pass: commit ─► push ─► create_pull_request ─► merge_pull_request
```

## Requirements

- Python 3.11+, `git`
- A GitHub login: `gh auth login` (the harness reuses that token) or `GITHUB_TOKEN` with `repo` scope
- An Anthropic API key: `export ANTHROPIC_API_KEY=...`, or put it in a `.env` file in the folder you
  run from (see `.env.example`; `.env` is git-ignored)

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

or `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.

## Use

Start by checking which issues it would pick up:

```bash
.venv/bin/pull-master https://github.com/you/your-repo --list
```

Fix them locally without touching GitHub (commits stay on local branches; patches are saved in the run folder):

```bash
.venv/bin/pull-master https://github.com/you/your-repo --dry-run
```

The full run (fix, test, open a PR, merge into the default branch):

```bash
.venv/bin/pull-master https://github.com/you/your-repo
```

Other common forms:

```bash
pull-master https://github.com/you/repo/issues/12          # one issue
pull-master you/repo --issue 12 --issue 15                 # specific issues
pull-master you/repo --label bug --limit 3                 # labelled issues
pull-master you/repo --no-merge                            # open PRs, leave merging to you
pull-master you/repo --test-cmd "make check" --setup-cmd "make deps"
```

| Option | Default | What it does |
|---|---|---|
| `--list` | | Show which issues would be worked on, then stop |
| `--dry-run` | | Fix and commit locally only. No push, PR, merge or comments |
| `--no-merge` | | Open PRs but don't merge them |
| `--base BRANCH` | repo default | Branch to merge into |
| `--merge-method` | `squash` | `squash`, `merge` or `rebase` |
| `--limit N` | 5 | Issues per run |
| `--allow-any-author` | off | Also take issues from people who aren't owners, members or collaborators |
| `--test-cmd`, `--setup-cmd` | detected | Override how tests run and how the environment is set up |
| `--max-attempts` | 3 | Test-and-retry rounds per issue |
| `--effort` | `high` | Claude effort: `low` … `max` |
| `--model` | `claude-opus-5-5` | Claude model |
| `--approve-commands` | off | Ask before each shell command the agent runs |
| `--comment-on-failure` | off | Comment on issues it couldn't fix or flagged for a person |
| `--mcp-command` | | Use a local MCP server over stdio instead of GitHub's hosted one |

Exit code: 0 when the run finished (some issues may be skipped or flagged for a person),
1 when an issue errored or a merge failed, 2 for setup problems (credentials, bad link).

## How it works

1. **GitHub over MCP.** The harness connects to GitHub's hosted MCP server
   (`https://api.githubcopilot.com/mcp/`) with your token. It uses `list_issues` and `issue_read`
   to pick work, and `create_pull_request` and `merge_pull_request` to land it. To run the
   server locally instead (for example with Docker Desktop running):
   `--mcp-command "docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server stdio"`.
2. **Sandbox.** The repo is cloned into `~/.pull_master/workspaces/<owner>__<repo>/repo`, a folder the
   harness owns and resets between issues. It never touches your own checkouts. Python projects
   get their own virtualenv next to the clone; Node projects get `npm ci`.
3. **Tests.** The test command is detected (`pytest`, `npm test`, `go test ./...`, `cargo test`,
   Maven/Gradle, `make test`) or given with `--test-cmd`. The suite runs once on the base branch first,
   so failures that were already there show up in the report and in the agent's instructions.
4. **The agent.** Claude (Opus 5.5 by default) gets four tools: a bash shell and a file editor
   (both limited to the clone), `run_tests`, and `submit_result`. It also gets a few read-only GitHub MCP
   tools (`issue_read`, `search_code`, `pull_request_read`, ...). It can't open PRs, merge, or comment.
5. **The gate.** When Claude submits a fix, the harness runs the tests itself. Only a passing run is
   committed, pushed, opened as a PR (`Fixes #N` in the body, so merging closes the issue) and merged.
   Failing output goes back to Claude for another try, up to `--max-attempts`.
6. **Reports.** Each run writes `report.md`, `report.json`, one transcript per issue (`.jsonl`) and
   the patch for each fix to `~/.pull_master/workspaces/<owner>__<repo>/runs/<timestamp>/`.

## Safety

"Sandbox" here means your own computer, so the harness tries to keep the agent from doing harm with it:

- **Untrusted issues are skipped by default.** Anyone can open an issue on a public repo, and the issue
  text goes to an agent that runs shell commands. Only issues from owners, members and collaborators are
  taken unless you pass `--allow-any-author`. The prompt also tells Claude to treat issue text as data.
- **No credentials in the agent's shell.** Variables that look like tokens or keys are removed, git
  credential helpers are switched off, `gh` is pointed at an empty config, and the SSH agent is dropped.
- **Only the harness pushes.** It pushes straight to the GitHub URL (not to a remote the agent could have
  changed), with git hooks disabled and `.git/config` restored from a snapshot first. The token exists
  only in the environment of that one git process.
- **The file editor can't leave the repository** (paths are resolved and checked; `.git` is off limits).
- **A few commands are blocked** (`git push`, `gh`, `sudo`, piping downloads into a shell, `rm -rf ~`).
  This is a speed bump, not a sandbox.

What it does not protect against: the agent's shell runs as you, so it can read your files and reach the
network. For untrusted repositories, run the harness inside a VM or container, or use `--approve-commands`.

## Notes

- If the base branch requires reviews or status checks, the merge fails and the PR is left open
  (status `merge_failed`).
- Refusal fallback is on: if a safety classifier declines a turn, the API retries it on a fallback model.
  Turn it off with `--no-fallback`.
- Issues are handled one after another, and each branch starts from the latest base branch, so
  later fixes include earlier merges.

## Development

```bash
.venv/bin/python -m pytest -q
```

The tests run the whole pipeline against a local bare git repo with a scripted model and a fake GitHub,
so they need no network or API key.
