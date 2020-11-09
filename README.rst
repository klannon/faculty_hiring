==============
Monte Carlo Methods for Faculty Hiring
==============

This repository contains some libraries to help with trying to model
the statistics of hiring faculty as well as a python notebook showing
the use of this code to answer a question about hiring strategy.


Description
===========

Being a particle physicist, I'm natrually inclined to answer
statistical questions with Monte Carlo methods.  This code attempts to
apply those methods to the question of faculty hiring.  The python
notebook included in the repository specifically tries to answer the
question, "What hiring strategy should one use when trying to hire
equitably for gender?"  (I will apologize in advance because this code
assumes gender is binary, so that the mathematical definition of
equity can be 50% of each gender.)

You can run the "hiring strategy" notebook interactively here: |logo|
You can run the "department" notebook interactively here: |logo2|
And you can run a notebook with different voting methods implement here: |logo3|

.. |logo| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/klannon/faculty_hiring/master?filepath=notebooks%2Fsearch_strategy.ipynb

.. |logo2| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/klannon/faculty_hiring/master?filepath=notebooks%2Fdepartment.ipynb

.. |logo3| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/klannon/faculty_hiring/master?filepath=notebooks%2Ftennessee_example.ipynb


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
