Re-running pyAFQ
~~~~~~~~~~~~~~~~
There are many circumstances where you may want to re-run pyAFQ on the same
data after changing some parameters or the data itself. This section describes how to
do that. You will want to use the pyAFQ `clobber` or `cmd_outputs` methods. These
are the same methods, `clobber` is just an alias for `cmd_outputs`. They are methods
attached to both `ParticipantAFQ` and `GroupAFQ`.

The `cmd_outputs` method allows you to perform operations on existing pyAFQ outputs,
which is particularly useful when you need to recalculate derivatives after changing
parameters. The most common use case is removing outputs so they can be regenerated,
or copying or moving outputs to a different location.

Method Parameters
-----------------
- ``cmd`` (str, optional): 
  The command to run on outputs. Default is 'rm' (remove). Other common commands might
  include 'cp' (copy) or 'mv' (move). Note that '-r' will be automatically added when
  operating on directories.

- ``dependent_on`` (str or None, optional):
  Specifies which derivatives to perform the command on:
  
  * ``None``: Perform on all outputs
  * ``"track"``: Only outputs dependent on tractography
  * ``"recog"``: Only outputs dependent on bundle recognition
  * ``"prof"``: Only outputs dependent on bundle profiling
  
  Default is None (all outputs). All "recog" outputs are dependent on "track" outputs,
  and all "prof" outputs are dependent on "recog" outputs. This means that if you
  specify "track", it will also remove all "recog" and "prof" outputs, for example.

- ``exceptions`` (list of str, optional):
  List of output names that should be excluded from the command. For example, you might
  want to remove all tractography-dependent files except the cleaned streamlines.
  Default is an empty list.

- ``suffix`` (str, optional):
  Additional command parts that should come after the filename. Default is an empty string.

Examples
--------

1. Remove all pyAFQ outputs to completely re-run the pipeline:

.. code-block:: python

   afq_object.clobber()

2. Remove only the outputs that depend on tractography (useful if you changed tracking parameters):

.. code-block:: python

   afq_object.clobber(dependent_on="track")

3. Move all outputs to a backup directory:

.. code-block:: python

   afq_object.cmd_outputs(
       cmd="mv",
       suffix="backup_directory/")
