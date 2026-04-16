"""Compatibility wrapper for the root distillation entrypoint."""

from distill_me_llm import *  # noqa: F401,F403


if __name__ == "__main__":
    from distill_me_llm import main

    main()
