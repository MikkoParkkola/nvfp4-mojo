# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability in nvfp4-mojo, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email **mikko.parkkola@iki.fi** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Impact assessment
4. Affected environment (MAX/Mojo version, platform, GPU)
5. Any suggested fix (optional)

You will receive an acknowledgment within 48 hours and an initial assessment within 7 days.

## Security Scope

In scope:

- Unsafe weight loading behavior, including unexpected file access or path handling bugs
- Kernel behavior that can corrupt process memory or expose data across tensors or requests
- Malformed model artifacts that trigger unsafe execution paths
- Build or dependency issues that compromise runtime integrity

Out of scope:

- Expected numerical drift from low-precision quantization
- Performance regressions without a security impact
- Issues that require local administrative access to the host beforehand

## Disclosure Policy

We follow coordinated disclosure:

1. Report privately by email
2. Confirm and assess within 7 days
3. Develop and test a fix
4. Release the fix and publish an advisory when appropriate
