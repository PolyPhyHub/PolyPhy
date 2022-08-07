# Contributing

Welcome to `polyphy` contributor\'s guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but [other kinds of contributions] are
also appreciated.

If you are new to using [git][other kinds of contributions] or have
never collaborated in a project previously, please have a look at
[contribution-guide.org][other kinds of contributions]. Other resources
are also listed in the excellent [guide created by
FreeCodeCamp][other kinds of contributions].

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, [Python
Software Foundation\'s Code of Conduct][other kinds of contributions] is
a good reference in terms of behavior guidelines.

# Issue Reports

If you experience bugs or general issues with `polyphy`, please have a
look on the [issue tracker][other kinds of contributions]. If you don\'t
see anything useful there, please feel free to fire an issue report.

::: tip
::: title
Tip
:::

Please don't forget to include the closed issues in your search.
Sometimes a solution was already reported, and the problem is considered
**solved**.
:::

New issue reports should include information about your programming
environment (e.g., operating system, Python version) and steps to
reproduce the problem. Please try also to simplify the reproduction
steps to a very minimal example that still illustrates the problem you
are facing. By removing other factors, you help us to identify the root
cause of the issue.

# Documentation Improvements

You can help improve `polyphy` docs by making them more readable and
coherent, or by adding missing information and correcting mistakes.

`polyphy` documentation uses [Sphinx][other kinds of contributions] as
its main documentation compiler. This means that the docs are kept in
the same repository as the project code, and that any documentation
update is done in the same way was a code contribution.

> e.g., [reStructuredText][other kinds of contributions] or
> [CommonMark][other kinds of contributions] with
> [MyST][other kinds of contributions] extensions.
>
> ::: tip
> ::: title
> Tip
> :::
>
> Please notice that the [GitHub web
> interface][other kinds of contributions] provides a quick way of
> propose changes in `polyphy`\'s files. While this mechanism can be
> tricky for normal code contributions, it works perfectly fine for
> contributing to the docs, and can be quite handy.
>
> If you are interested in trying this method out, please navigate to
> the `docs` folder in the source
> [repository][other kinds of contributions], find which file you would
> like to propose changes and click in the little pencil icon at the
> top, to open [GitHub\'s code editor][other kinds of contributions].
> Once you finish editing the file, please write a message in the form
> at the bottom of the page describing which changes have you made and
> what are the motivations behind them and submit your proposal.
> :::

When working on documentation changes in your local machine, you can
compile them using \_:

    tox -e docs

and use Python\'s built-in web server for a preview in your web browser
(`http://localhost:8000`):

    python3 -m http.server --directory 'docs/_build/html

  [other kinds of contributions]: