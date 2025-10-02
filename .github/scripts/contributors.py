import os
import re

import requests

OWNER = os.getenv("GITHUB_REPOSITORY").split("/")[0]
REPO = os.getenv("GITHUB_REPOSITORY").split("/")[1]
TOKEN = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {TOKEN}"}


def paginated_get(url):
    results = []
    while url:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(
                f"Error: Failed to fetch {url} (status code {resp.status_code})\n"
                f"Response: {resp.text}"
            )
            break
        data = resp.json()
        results.extend(data)
        # Pagination: look for 'next' link
        url = None
        if "link" in resp.headers:
            links = resp.headers["link"].split(",")
            for link in links:
                if 'rel="next"' in link:
                    url = link[link.find("<") + 1 : link.find(">")]
                    break
    return results


def fetch_contributors():
    # Commits
    commit_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/contributors?per_page=100"
    )
    commit_data = paginated_get(commit_url)
    committers = {user.get("login") for user in commit_data if user.get("login")}

    # Closed Issues
    labels = ["bug", "enhancement", "discussion"]

    issues_data = []
    for label in labels:
        issues_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=closed&labels={label}&per_page=100"
        issues_data += paginated_get(issues_url)

    print(len(issues_data))

    issuers = {
        issue.get("user", {}).get("login")
        for issue in issues_data
        if "pull_request" not in issue and issue.get("user")
    }

    # Merged Pull Requests
    prs_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=closed&per_page=100"
    )
    prs_data = paginated_get(prs_url)
    pr_authors = {
        pr.get("user", {}).get("login")
        for pr in prs_data
        if pr.get("merged_at") and pr.get("user")
    }

    # PR reviews
    reviewers = set()
    for pr in prs_data:
        if pr.get("merged_at"):
            reviews_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr['number']}/reviews?per_page=100"
            reviews = paginated_get(reviews_url)
            for r in reviews:
                user = r.get("user")
                if user and user.get("login"):
                    reviewers.add(user["login"])

    # Remove None values if any
    committers = sorted(committers)
    issuers = sorted(issuers)
    pr_authors = sorted(pr_authors)
    reviewers = sorted(reviewers)

    print(
        f"Found {len(committers)} committers, {len(issuers)} issuers, {len(pr_authors)} PR authors, {len(reviewers)} reviewers."
    )
    print("Committers:", committers)
    print("Issuers:", issuers)
    print("PR Authors:", pr_authors)
    print("Reviewers:", reviewers)

    return committers, issuers, pr_authors, reviewers


def gh(username):
    return f"[@{username}](https://github.com/{username})"


def append_to_readme(committers, issuers, pr_authors, reviewers):
    section_header = "## Contributors"
    table = (
        "\n## Contributors\n\n"
        "`ondil` was developed by Simon Hirsch, Jonathan Berrisch and Florian Ziel. \n"
        "We're grateful for contributions below (sorted alphabetically by GitHub username).\n\n"
        "| Contribution | GitHub Users |\n"
        "|-------------------|--------------|\n"
        "| Code | " + ", ".join(gh(u) for u in committers) + " |\n"
        "| Reported (closed) Issues | " + ", ".join(gh(u) for u in issuers) + " |\n"
        "| Merged PRs | " + ", ".join(gh(u) for u in pr_authors) + " |\n"
        "| PR Reviews | " + ", ".join(gh(u) for u in reviewers) + " |\n"
    )

    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        if section_header in content:
            # Replace existing section
            pattern = r"\n## Contributors\n\n\|.*?\|\n(?:\|.*?\|\n)*"
            new_content = re.sub(pattern, table, content, flags=re.DOTALL)
            if new_content == content:
                # Fallback: replace from header to next header or EOF
                pattern2 = r"\n## Contributors.*?(?=\n##|\Z)"
                new_content = re.sub(pattern2, table, content, flags=re.DOTALL)
            content = new_content
        else:
            content += table
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(table)


if __name__ == "__main__":
    print("Owner:", OWNER)
    print("Repo:", REPO)
    c, i, p, r = fetch_contributors()
    append_to_readme(c, i, p, r)
