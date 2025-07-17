.. _modeling:

Modeling White Matter
=====================

In order to make inferences about white matter tissue properties, we use a
variety of models. The models are fit to the data in each voxel and the
parameters of the model are used to interpret the signal.

For an interesting perspective on modeling of tissue properties from diffusion
MRI data, please refer to a recent paper by Novikov and colleagues
:cite:`Novikov2018`.

`This page <https://tractometry.org/pyAFQ/reference/methods.rst>` includes
a list of the model parameters that are accessible through the
:class:`AFQ.api.group.GroupAFQ` and :class:`AFQ.api.participant.ParticipantAFQ`
objects.
