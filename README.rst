================
Python MagnetRun
================


.. image:: https://img.shields.io/pypi/v/python_magnetrun.svg
        :target: https://pypi.python.org/pypi/python_magnetrun

.. image:: https://img.shields.io/travis/Trophime/python_magnetrun.svg
        :target: https://travis-ci.com/Trophime/python_magnetrun

.. image:: https://readthedocs.org/projects/python-magnetrun/badge/?version=latest
        :target: https://python-magnetrun.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Python MagnetRun contains utils to view and analyse Magnet runs


* Free software: MIT license
* Documentation: https://python-magnetrun.readthedocs.io.


Features
--------

* Load txt, cvs and tdms files from control/monitoring system
* Extract field(s)
* Plot field(s) vs time
* Plot field vs field

TODO
--------

* tdms to pandas see https://nptdms.readthedocs.io/en/stable/apireference.html
* check addData formula with help of python pyparsing??
* export Data to prettytables? tabular? cvs2md?
* how to pass option to matplotlib within plotData()?*args PARAMETER TO MAKE OPTIONAL ARGUMENTS?
* add support for origin files (for B experimental profil) - use labplot?? liborigin?? python bindings??
* get MagnetRun files from control/monitoring system??
* how to add columns comming from freesteam, for instance like rho, cp...??
* for MagnetRun add missing fields (U1, Pe1, Tout1, U2 ...], --missing, --nhelices - see txt2csv.py
	    
	    
	    
	    
	    
   
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
