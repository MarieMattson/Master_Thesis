## Marie's Master Thesis Repo

This is where I keep my code for my thesis. In this file I also put notes, just for myself, to remember commands.

### UV
Virtual enviroment. It looks like a normal folder. If *uv* is added before a command, it is done in the virtual envirioment. This is how it is initailised:
> uv init thesis

It automatically includes *pyproject.toml*, which shows my dependencies.

I have an alias for running python, instead of:
> uv run python file.py
I can use:
> up file.py

This command was used to create this alias:
> alias up="uv run python"

### Git commands
Checks status.
> git status

Adds to be ready for commit.
> git add hello.txt

Commits, message need be included
> git commit -m "hello"

Pushes to github so it is visible online
> git push

This command is equivalent of *git config --global alias.g "log --oneline --all --graph"*. It shows commits in a list
> git g