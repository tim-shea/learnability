# learnability
This is a set of python scripts for various STDP learning experiments.

## prereqs
These scripts require scipy and brian2 and are developed for use with Python 2.7. The easiest way to run
any script in this repo is to install the anaconda scipy distribution and then execute `pip install brian2`.

Most scripts require matplotlib and therefore depend upon a graphical backend.

Several of the scripts define library like functions used in other. When executed directly these will
run some functionality tests.

Finally, some scripts may invoke weave or cython functions--directly or indirectly--and therefore require
a properly configured installation of these libraries as well. This is generally easiest to accomplish from
a linux OS.
