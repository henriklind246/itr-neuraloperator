"""Daily GitHub commit review: fetches today's commits, summarizes via Claude, posts a GitHub Issue."""

from __future__ import annotations

import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

import anthropic
import requests

CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOTAL_DIFF_CHARS = 60_000
MAX_PER_FILE_CHARS = 3_000
SKIP_EXTENSIONS = {".lock", ".min.js", ".min.css", ".npy", ".pt", ".png", ".jpg", ".jpeg", ".gif", ".woff", ".woff2"}


def _get_config() -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set.")
    repo = os.environ.get("REPO_FULL_NAME")
    if not repo:
        raise RuntimeError("REPO_FULL_NAME environment variable is not set.")
    timezone = os.environ.get("REVIEW_TIMEZONE", "UTC")
    return {"api_key": api_key, "token": token, "repo": repo, "timezone": timezone}


def _github_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _get_today_commits(token: str, repo: str, timezone: str) -> list[dict]:
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    start_of_day = datetime.combine(now.date(), time.min, tzinfo=tz)
    since = start_of_day.isoformat()
    until = now.isoformat()
    headers = _github_headers(token)

    # List all branches
    branches_url = f"https://api.github.com/repos/{repo}/branches"
    branches = []
    page = 1
    while True:
        resp = requests.get(branches_url, headers=headers, params={"per_page": 100, "page": page})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        branches.extend(batch)
        page += 1

    # Fetch commits from each branch, deduplicate by SHA
    seen_shas: set[str] = set()
    all_commits: list[dict] = []
    for branch in branches:
        url = f"https://api.github.com/repos/{repo}/commits"
        params = {"sha": branch["name"], "since": since, "until": until, "per_page": 100}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for commit in resp.json():
            if commit["sha"] not in seen_shas:
                seen_shas.add(commit["sha"])
                all_commits.append(commit)

    return all_commits


def _fetch_commit_diffs(token: str, repo: str, commits: list[dict]) -> list[dict]:
    enriched = []
    for commit in commits:
        sha = commit["sha"]
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        resp = requests.get(url, headers=_github_headers(token))
        resp.raise_for_status()
        data = resp.json()

        files_summary = []
        for f in data.get("files", []):
            entry = {
                "filename": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
            }
            patch = f.get("patch", "")
            if any(f["filename"].endswith(ext) for ext in SKIP_EXTENSIONS):
                entry["patch"] = f"[binary/generated file, +{f['additions']}/-{f['deletions']}]"
            elif len(patch) > MAX_PER_FILE_CHARS:
                entry["patch"] = (
                    patch[:MAX_PER_FILE_CHARS]
                    + f"\n[... truncated, +{f['additions']}/-{f['deletions']} total]"
                )
            else:
                entry["patch"] = patch
            files_summary.append(entry)

        # For merge commits with many files, skip redundant diffs
        message = data["commit"]["message"]
        if message.startswith("Merge") and len(files_summary) > 20:
            for entry in files_summary:
                entry["patch"] = f"[merge commit — +{entry['additions']}/-{entry['deletions']}]"

        enriched.append({
            "sha": sha[:8],
            "message": message,
            "author": data["commit"]["author"]["name"],
            "date": data["commit"]["author"]["date"],
            "files": files_summary,
        })
    return enriched


def _apply_global_diff_budget(commits: list[dict]) -> list[dict]:
    total = sum(len(e["patch"]) for c in commits for e in c["files"])
    if total <= MAX_TOTAL_DIFF_CHARS:
        return commits

    # Progressively strip longest patches until under budget
    all_entries = [(c, e) for c in commits for e in c["files"]]
    all_entries.sort(key=lambda x: len(x[1]["patch"]), reverse=True)

    for _, entry in all_entries:
        if total <= MAX_TOTAL_DIFF_CHARS:
            break
        old_len = len(entry["patch"])
        entry["patch"] = (
            f"[diff omitted for size — {entry['filename']}, "
            f"{entry['status']}, +{entry['additions']}/-{entry['deletions']}]"
        )
        total -= old_len - len(entry["patch"])
    return commits


def _build_system_prompt(repo: str, date_str: str) -> str:
    return f"""\
You are a technical reviewer analyzing a day's worth of commits to a GitHub repository ({repo}).

Provide your analysis in the following format:

## Daily Commit Review — {date_str}

### What was done
A plain-language summary of all work performed today. Describe what was built, changed, \
or fixed in terms a technical manager would understand. Group related commits together.

### Complexity Assessment
Rate the complexity of today's work on a scale of 1-5:
1 = Trivial (typo fixes, comment changes, config tweaks)
2 = Simple (straightforward additions, minor refactors)
3 = Moderate (new features with some design decisions, meaningful refactors)
4 = Complex (architectural changes, algorithm implementations, multi-component features)
5 = Highly Complex (novel algorithms, deep mathematical/scientific work, major system redesigns)

Justify your rating in 1-2 sentences.

### Volume Assessment
Rate the volume of work on a scale of 1-5:
1 = Minimal (1-2 trivial commits)
2 = Light (a few small changes)
3 = Moderate (several meaningful commits or one substantial piece of work)
4 = Heavy (multiple substantial changes across the codebase)
5 = Very Heavy (extensive work touching many components)

Justify your rating in 1-2 sentences.

### Verdict
One sentence: was this a good day's work? Consider both complexity and volume together.

### Commit Details
A brief bullet point for each commit with its SHA prefix."""


def _build_user_message(commits: list[dict], repo: str, date_str: str) -> str:
    parts = [f"Here are today's commits to {repo} on {date_str}:\n"]
    for c in commits:
        parts.append(f"\n--- Commit {c['sha']} by {c['author']} at {c['date']} ---")
        parts.append(f"Message: {c['message']}\n")
        parts.append("Files changed:")
        for f in c["files"]:
            parts.append(f"  {f['filename']} ({f['status']}, +{f['additions']}/-{f['deletions']})")
        parts.append("\nDiffs:")
        for f in c["files"]:
            if f["patch"]:
                parts.append(f"\n### {f['filename']}")
                parts.append(f["patch"])
    return "\n".join(parts)


def _call_claude(system_prompt: str, user_message: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def _ensure_label(token: str, repo: str) -> None:
    url = f"https://api.github.com/repos/{repo}/labels"
    payload = {"name": "daily-review", "color": "7057ff", "description": "Automated daily commit review"}
    resp = requests.post(url, headers=_github_headers(token), json=payload)
    # 422 = already exists, which is fine
    if resp.status_code not in (201, 422):
        print(f"Warning: could not create label (status {resp.status_code})")


def _create_github_issue(token: str, repo: str, title: str, body: str) -> str:
    _ensure_label(token, repo)
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = {"title": title, "body": body, "labels": ["daily-review"]}
    resp = requests.post(url, headers=_github_headers(token), json=payload)
    resp.raise_for_status()
    return resp.json()["html_url"]


def main() -> None:
    config = _get_config()
    tz = ZoneInfo(config["timezone"])
    date_str = datetime.now(tz).strftime("%Y-%m-%d")

    print(f"Fetching commits for {date_str} (timezone: {config['timezone']})...")
    commits = _get_today_commits(config["token"], config["repo"], config["timezone"])

    if not commits:
        print("No commits found for today. Skipping review.")
        return

    print(f"Found {len(commits)} commit(s). Fetching diffs...")
    enriched = _fetch_commit_diffs(config["token"], config["repo"], commits)
    enriched = _apply_global_diff_budget(enriched)

    system_prompt = _build_system_prompt(config["repo"], date_str)
    user_message = _build_user_message(enriched, config["repo"], date_str)
    print(f"Prompt size: {len(system_prompt) + len(user_message):,} chars. Calling Claude...")

    review = _call_claude(system_prompt, user_message, config["api_key"])

    print("\n" + "=" * 60)
    print(review)
    print("=" * 60 + "\n")

    title = f"Daily Commit Review — {date_str}"
    issue_url = _create_github_issue(config["token"], config["repo"], title, review)
    print(f"Created issue: {issue_url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
