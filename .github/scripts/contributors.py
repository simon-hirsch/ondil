import os

import requests

OWNER = os.getenv("GITHUB_REPOSITORY").split("/")[0]
REPO = os.getenv("GITHUB_REPOSITORY").split("/")[1]
TOKEN = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {TOKEN}"}


def fetch_contributors():
    # Commits
    commit_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contributors"
    commit_data = requests.get(commit_url, headers=headers).json()
    committers = {user["login"] for user in commit_data if "login" in user}

    # Closed Issues
    issues_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=closed&per_page=100"
    )
    issues_data = requests.get(issues_url, headers=headers).json()
    issuers = {
        issue["user"]["login"] for issue in issues_data if "pull_request" not in issue
    }

    # Merged Pull Requests
    prs_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=closed&per_page=50"
    )
    prs_data = requests.get(prs_url, headers=headers).json()
    pr_authors = {pr["user"]["login"] for pr in prs_data if pr.get("merged_at")}

    # PR reviews
    reviewers = set()
    for pr in prs_data:
        if pr.get("merged_at"):
            reviews_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr['number']}/reviews"
            reviews = requests.get(reviews_url, headers=headers).json()
            for r in reviews:
                if "user" in r and r["user"]:
                    reviewers.add(r["user"]["login"])

    return committers, issuers, pr_authors, reviewers


def write_file(committers, issuers, pr_authors, reviewers):
    with open("CONTRIBUTORS.md", "w", encoding="utf-8") as f:
        f.write("# Contributors\n\n")
        f.write("### Code\n")
        for u in sorted(committers):
            f.write(f"- @{u}\n")
        f.write("\n### Closed Issues\n")
        for u in sorted(issuers):
            f.write(f"- @{u}\n")
        f.write("\n### Merged PRs\n")
        for u in sorted(pr_authors):
            f.write(f"- @{u}\n")
        f.write("\n### PR Reviews\n")
        for u in sorted(reviewers):
            f.write(f"- @{u}\n")


if __name__ == "__main__":
    c, i, p, r = fetch_contributors()
    write_file(c, i, p, r)
