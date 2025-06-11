# Marie's Master Thesis Repo

This is where I keep my code for my thesis. In this file I also put notes for myself. The pipelines I created are in the directory llm_retrieval. They contain a FAISS vector index pipeline, KGRAG pipeline as well as a KGRAG pipeline using lanchain, which works but was never used in experiments. Data is created and indexed in the directory "data_import". This is also where data is analysed in notebooks (used for explaining data in the thesis). This folder contains both large and small datasets, as well as their vector indices (not present in git due to size).

In addition, there is an evaluation folder, containing both LLM based evaluation and numeral evaluation (based on things like matching metadata and speech_ID). There is also the folder for running the experiments.

The thesis is currenlty being graded. When it is graded, it will be published to DIVA, link will come.

## Notes about commands and frameworks used

### Neo4j
Before you can use Neo4j on the wsl, you must install it on the wsl. Follow this tutoriil (only the chapters "How to install Neo4j" and "How to test connection"):
> https://www.techrepublic.com/article/how-to-install-neo4j-ubuntu-server/ 

To get it to work, open this config file:
> sudo nano /etc/neo4j/neo4j.conf

And make sure this line is uncommented:

```ini
# With default configuration Neo4j only accepts local connections.

# To accept non-local connections, uncomment this line:

server.default_listen_address=0.0.0.0
```

To make the Langchain connection work, this line also has to changed and uncommented in the config:
```ìni
#dbms.security.procedures.unrestricted=my.extensions.example,my.procedures.*

Must be changed to 
dbms.security.procedures.unrestricted=apoc.coll.*,apoc.load.*,gds.*,apoc.*

And the line below it, which likely says :
dbms.security.procedures.allowlist=apoc.coll.*,apoc.load.*,gds.*
Must be changed to:
dbms.security.procedures.allowlist=apoc.coll.*,apoc.load.*,gds.*,apoc.*
```

> sudo systemctl restart neo4j.service 

The neo4j server should start when wsl is started. However, if it does not, try this:
> sudo systemctl start neo4j.service 

To acces the GUI of neo4j, the easiest thing is to open it in browser. Use this url:
> http://localhost:7474/

### UV
This is the virtual enviroment used. These comments are mostly for myself if I want to go back and run my code. This is how it is initailised:
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