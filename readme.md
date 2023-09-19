# Quantitative Portfolio Tool

The idea behind this project is to create a self-sustaining framework for managing and iterating on new and existing quant strategies that will make up our portfolio.

## Team

1. Anthony Givans

    > Rising Junior, studying Math and CS with a minor in Astrophysics. Does Computer Vision Research and is really interested in applying Neural Nets to modern day problems. Currently working on making a Math and Physics LLM that takes a Math (or Physics) question and provides a step-by-step approach to solving said question. Also worked at Google (Summer 2023). You may contact me at <anthonygivans876@gmail.com> or <anthonygivans@miami.edu>.

2. Yash Patel

    > Rising Sophomore, studying Math, CS, and Physics. Doing research on Classifiers, particularly Deep Neural Nets, at an REU, a NSF Funded Research Fellowship. Observing Inter/Intra-Class Boundaries in the latent space, with potential applications to evaluate the robustness of a classifier without prior training data or test data. Plan to develop a mock betting simulator, akin to those used for Quantiative Trading interviews, to develop strategy.


3. Matthew Mione

    > Sophomore, studying Math and CS. Interested in doing pure math research in fields of probability theory and mathematical modeling of real-world scenarios. Fun fact about me: I have a cookie monster onesie, and I like to play piano. You may contact me at <matthewmione@gmail.com> or <mxm194032@miami.edu>. 
   
4. Sabian Saliasi

    > Rising Junior, studying Math and CS. Presently doing research on Machine Learning methods within Quantitative Finance and working as a Software Engineer in a start-up environment as a Machine Learning and Software Engineer at AI Academy, pushing to provide resources in Math, CS, ML, etc, to students from under-represented backgrounds.

## How to Contribute

Note: these instructions assume a working knowledge of both `git` and `github` and that you also have `git` already installed on your machine

* Firstly, `fork` this repository

> Look at the top right-hand corner of the page. You should see the option to `fork` the repo

* You will then clone the forked repo on your machine. You should see a green `code` button on the (`forked`) repo's page. Click that button, then copy the `.git` link that is present. Next, find a folder on your local machine where you are comfortable storing this code. I recommend `Desktop` but if you have an existing file structure on your machine, feel free to use it. Once you have decided on where you want to store the code, enter the following command in your command prompt (eg, cmd prompt, bash, git bash, ubuntu, etc):

> git clone "the link you copied earlier that ended in `.git`"

At this point, you should have seen some output in your terminal and some files should have populated your directory. That's great!

* Next, we have some housekeeping stuff to go over:

We are using poetry to track our dependencies, which will make all our lives easier in the long-run. We already took care of this, so run this command in your terminal:

For Linux, macOS, or WSL do the following:

> curl -sSl <https://install.python-poetry.org> | python3 -

You will need to add this to your PATH. Type `vim .bashrc` and add the command referenced to the end of the file. You will notice that you can't use your mouse and that you can't type anything. Press `i` to enter `insert mode`. Navigate to the end of the file and add the command referenced after the `poetry` install. Note that you are currently in the infamous `vim`, where a lot of people get stuck and cannot leave. When you're done, hit `esc` to exit `insert` mode, then `:` followed by `wq` and `enter` (in that order). Great, now you have successfully entered and exited `vim`. Now, restart your terminal (close and reopen) and proceed.

If you are using Windows Powershell, follow this [link](https://python-poetry.org/docs/#installing-with-the-official-installer) to get the instructions

Next, run

> poetry install

This command should have created a new environment and installed all the dependencies that you need to run the project in said environment. This command uses both the `pyproject.toml` and `poetry.lock` to setup the environment. In order to enter that environment, run

> poetry shell

Here, you should be able to run the files and test the code. We have a bit more housekeeping before we are done. Let's setup. Run the following:

> poetry add pre-commit

You may have noticed that we used `poetry` to install our dependency instead of `pip`. This is because `poetry` manages our dependencies and ensures that these dependencies are compatible. Also, `poetry` writes these chnages to the `poetry.lock` file so that the other persons working on the project will be able to download these dependencies as well. Whenever you have a new dependency, use `poetry add dependency` instead of `pip install dependency`. Lastly, run

> pre-commit install

There is a `.precommit-config.yaml` that specifies some interesting tasks that must be done (automatically) before commiting your code. These ensure that your code is bug free and conforms to the PEP8 standard. It also gives you a `coverage` summary, which shows you how much of your code is tested. Since these are early days, we are aiming for a **\>90%** coverage percentage.

**For Mac users:** if the command above doesn't work and gives you a "command not found error", run the following commands then rerun `pre-commit install`

> brew install pre-commit
> poetry update package

As a quick example, run the following command

> poetry run tox

or, simply

> tox

This should run all the tests in the `/tests` folder and also provide and `coverage` summary for each run.

Now, let's talk about how you submit your code. The main branch is locked, meaning you can't directly make changes to it. You will have to create a new branch, using `git`, with the following command

> git checkout -b "name of branch" (without the quotes)

This will create a new branch, then switch to that branch, which is great!

After this, commit your changes to the branch and then submit your **PR** (Pull Request) for review. In order to push your code to your remote branch and **NOT** the `main` branch, use the following command whenever you are pushing code from a new branch for the work time:

> git push --set-upstream origin name_of_branch

On the right-hand side of the PR page, you will see an option for **Reviewers**, where you can request to assign two people to review your **PR**. Doing this will expedite the time it takes to get approved!

### GOOD LUCK! :)
