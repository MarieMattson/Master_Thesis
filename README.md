## Marie's Master Thesis Repo

This is where I keep my code for my thesis. In this file I also put notes, just for myself, to remember commands.

### Neo4j
Before you can use Neo4j on the wsl, you must install it on the wsl. Follow this tutorail (only the chapters "How to install Neo4j" and "How to test connection"):
> https://www.techrepublic.com/article/how-to-install-neo4j-ubuntu-server/ 

To get it to work, open this config file:
> sudo nano /etc/neo4j/neo4j.conf

And make sure this line is uncommented:

```ini
# With default configuration Neo4j only accepts local connections.

# To accept non-local connections, uncomment this line:

server.default_listen_address=0.0.0.0
```

> sudo systemctl restart neo4j.service 

The neo4j server should start when wsl is started. However, if it does not, try this:
> sudo systemctl start neo4j.service 

To acces the GUI of neo4j, the easiest thing is to open it in browser. Use this url:
> http://localhost:7474/ 

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

To add a library, instead of pip install, we use
> uv add library

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