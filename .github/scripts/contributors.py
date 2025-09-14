import os

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
    issues_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=closed&per_page=100"
    )
    issues_data = paginated_get(issues_url)
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
    committers.discard(None)
    issuers.discard(None)
    pr_authors.discard(None)
    reviewers.discard(None)

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
