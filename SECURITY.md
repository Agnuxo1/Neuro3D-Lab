# Security Policy

## Supported versions

Only the current version on the `master` branch (0.1.0) receives security fixes.

## Reporting a vulnerability

Do not report security vulnerabilities in public issues, pull requests or discussions.

Report them privately through GitHub security advisories:

https://github.com/Agnuxo1/Neuro3D-Lab/security/advisories/new

Include:

- the affected version or commit,
- steps to reproduce and the impact you expect,
- a proof of concept, if you can share it safely with the maintainer.

Acknowledgement is best effort. The maintainer will keep you informed of progress and will credit you in the advisory if you wish.

## Scope

The core library (`src/optlab`) does not access the network. It writes files only to paths that the caller supplies (for example `.npy` outputs and JSONL run logs). The Blender add-on writes bench files to a path chosen by the user. Vulnerabilities in third-party dependencies (numpy, scipy, Blender) should be reported to their upstream projects, although notes about how this project uses them are welcome.
