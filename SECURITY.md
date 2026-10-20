# Security policy

## Reporting a vulnerability

Report security issues privately through GitHub's vulnerability reporting for this repository: open the **Security** tab and choose **Report a vulnerability**, or go directly to <https://github.com/sgl-project/sglang-torchtpu/security/advisories/new>.

Do not open a public issue, pull request, or discussion for a security problem, and do not post it in Slack. Public reports can be exploited before a fix exists.

A useful report contains:

- the component (plugin code under `sglang_tpu/`, a launch recipe, a CI workflow, a published image);
- the sglang-torchtpu commit or release, and the torch_tpu and libtpu versions;
- the chip generation and topology, and the launch command;
- steps to reproduce, and what an attacker gains (code execution, data from another request, denial of service, credential exposure).

## What to expect

- Acknowledgement within three business days.
- An initial assessment within ten business days, with a severity and whether we consider it in scope.
- A fix or mitigation on a timeline agreed with you, typically within 90 days for confirmed issues, faster for anything that exposes credentials or crosses request boundaries.
- Credit in the advisory and the release notes, unless you prefer not to be named.

We ask that you give us that window before disclosing publicly, and that you do not access, modify, or exfiltrate data that is not yours while testing.

## Supported versions

| Version | Security fixes |
| --- | --- |
| `main` | Yes |
| Latest release | Yes |
| Older releases | No; upgrade to the latest release |

Until the first tagged release, `main` is the only supported line.

## Scope

In scope:

- Code in this repository: the TPU platform and kernels under `sglang_tpu/`, the fork's TPU-specific changes under `python/`, launch and benchmark recipes, and tests.
- CI workflows and tooling under `.github/` and `scripts/`.
- Container images published by this project.

Out of scope here, with where to report instead:

- SGLang engine code that behaves the same on GPU: [sgl-project/sglang security policy](https://github.com/sgl-project/sglang/security/policy).
- torch_tpu, libtpu, JAX, and other Google TPU software: the respective project's reporting channel.
- Third-party model checkpoints and tokenizers loaded from the Hugging Face Hub or elsewhere.
- Deployments of this software operated by others.

Vulnerabilities in a pinned dependency that affect this project are in scope for the fix, meaning a pin bump and an advisory, even though the root cause is reported upstream.

## Deployment notes

SGLang-torchtpu serves an HTTP API with no built-in authentication, like SGLang on other hardware. Run it behind an authenticating proxy, never exposed directly to untrusted networks. Loading a model checkpoint executes its configuration and, for some formats, its code; only serve checkpoints you trust. Multi-host launches exchange data over the host network between workers; keep that network private.

## Preferences

No bug bounty is offered. We follow coordinated disclosure and publish a GitHub security advisory for every confirmed vulnerability with a CVE where one applies.
