# Contributing

Thanks for your interest in improving nvfp4-mojo.

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make a focused change
4. Run the relevant tests before opening a pull request

## Development Setup

This project targets Mojo/MAX development environments.

Run the test suite with the installed Mojo toolchain:

```bash
mojo run run_tests.mojo
```

If you manage the environment through `pixi`, run the same command inside that environment.

## Contribution Guidelines

- Keep changes scoped to a clear kernel, loader, or test improvement
- Add or update tests for behavior changes
- Update the README when file layout, setup, or usage instructions change
- Include platform and toolchain details when reporting bugs or performance regressions
