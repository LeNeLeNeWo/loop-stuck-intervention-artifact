# Open-Source Anonymization Steps

1. Create a GitHub repository with a neutral name, for example `loop-stuck-intervention-artifact`.
2. Copy or push the contents of `release_artifacts/open_source_staging/` to that repository.
3. Before committing, use an anonymous local Git identity:

```bash
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.com"
```

4. Commit the staged artifact and record the fixed commit hash used for submission.
5. Go to `https://anonymous.4open.science/`.
6. Authenticate with GitHub.
7. Select the repository and the fixed commit hash.
8. Add redaction terms for all author names, emails, handles, institutions, local path fragments, lab names, and private project identifiers.
9. Enable display for Markdown, PDF, and image files if the service offers those options.
10. Generate the anonymous repository link in the form `https://anonymous.4open.science/r/<anonymous-id>/`.
11. Put the anonymous link in the paper and submission form.
12. Optionally upload `LoopSense_Paper1_Artifact.zip` as supplementary material if the conference system allows.
13. After acceptance, replace the anonymous link with the final public GitHub or archival DOI link.

Important warnings:

- Anonymous GitHub mainly anonymizes textual files shown through the service.
- Binary files and zip contents must already be clean before upload.
- Do not rely on the service to remove identities from compressed artifacts.
- Files larger than service limits may not render in the anonymous web view.
- Do not push raw trajectories, raw live workspaces, Docker caches, `.env` files, API keys, or local path-bearing logs.
