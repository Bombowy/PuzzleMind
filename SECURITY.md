# Security Policy

## Supported versions

LogicForge is pre-alpha software. Security fixes are applied only to the latest
revision of the `main` branch until a stable release policy is published.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Contact the repository
owner privately through their GitHub profile with a description, reproduction
steps, affected revision, and potential impact. Avoid including unnecessary private
screenshots or credentials.

The project will acknowledge a report when it is reviewed, coordinate validation
and remediation privately, and credit reporters who want attribution. No response
time guarantee is offered before the project has a dedicated security team.

## Sensitive areas

Image decoders, untrusted plugin discovery, artifact paths, screenshot privacy, and
desktop automation will require explicit threat models before implementation.
Automation must remain disabled by default and must include a user-controlled
emergency stop before it can emit real input events.
