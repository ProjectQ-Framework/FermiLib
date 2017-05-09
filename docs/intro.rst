.. _intro:

Tutorial
========

.. toctree::
   :maxdepth: 2	

Getting started with FermiLib
-----------------------------

Installing FermiLib requires pip. Make sure that you are using an up-to-date version of it. Then, install FermiLib, by running

.. code-block:: bash

	python -m pip install --pre --user fermilib

Alternatively, clone/download `this repo <https://github.com/ProjectQ-Framework/FermiLib>`_ (e.g., to your /home directory) and run

.. code-block:: bash

	cd /home/fermilib
	python -m pip install --pre --user .

This will install FermiLib and all its dependencies automatically. In particular, FermiLib requires `ProjectQ <https://projectq.ch>`_ . It might be useful to install ProjectQ separately before installing FermiLib as it might require setting some manual options such as, e.g., a C++ compiler. Please follow the `ProjectQ installation <https://projectq.ch/code-and-docs/>`_ instructions. FermiLib is compatible with both Python 2 and 3.


Basic FermiLib example
----------------------

To see a basic example with both fermionic and qubit operators as well as whether the installation worked, try to run the following code.

.. code-block:: python

	from fermilib.ops import FermionOperator, hermitian_conjugated
	from fermilib.transforms import jordan_wigner, bravyi_kitaev
	from fermilib.utils import eigenspectrum
	
	# Initialize an operator.
	fermion_operator = FermionOperator('2^ 0', 3.17)
	fermion_operator += hermitian_conjugated(fermion_operator)
	print(fermion_operator)
		
	# Transform to qubits under the Jordan-Wigner transformation and print its spectrum.
	jw_operator = jordan_wigner(fermion_operator)
	jw_spectrum = eigenspectrum(jw_operator)
	print(jw_operator)
	print(jw_spectrum)
	
	# Transform to qubits under the Bravyi-Kitaev transformation and print its spectrum.
	bk_operator = bravyi_kitaev(fermion_operator)
	bk_spectrum = eigenspectrum(bk_operator)
	print(bk_operator)
	print(bk_spectrum)


This code creates the fermionic operator :math:`a^\dagger_2 a_0` and adds its Hermitian conjugate :math:`a^\dagger_0 a_2` to it. It then maps the resulting fermionic operator to qubit operators using two transforms included in FermiLib, the Jordan-Wigner and Bravyi-Kitaev transforms. Despite the different representations, these operators are iso-spectral. The example also shows some of the intuitive string methods included in FermiLib.

Further examples can be found in the docs (`Examples` in the panel on the left) and in the FermiLib examples folder on `GitHub <https://github.com/ProjectQ-Framework/FermiLib/tree/master/examples/>`_.
