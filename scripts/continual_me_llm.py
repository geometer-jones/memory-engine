"""Compatibility wrapper for the root continual-learning entrypoint."""

from continual_me_llm import *  # noqa: F401,F403


if __name__ == "__main__":
    from continual_me_llm import main

    main()
