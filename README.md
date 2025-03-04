# Machine Learning Project - SVM
Project for the "Optimization Methods for Machine Learning" course.

Master's degree course in "Business Intelligence and Analytics" (Management engineering's curricula).

## Project Description
The project, developed in `Python`, involves the implementation of various approaches and techniques regarding `Support Vector Machines` in machine learning, putting particular emphasis on the optimization side of the developement. The goal was to build a classifier distinguishing and classifying handwrittend digits.

The 4 implementations of the project required developing concepts like: Hard/Soft SVM, Dual SVM, grid search (with kfold cross validation), SVM light and worker's selection, Kernel trick, Sequential Minimal Optimization with Most violating pair (MVP decomposition method), multi-class SVM ecc.

All the work conducted had the aim of optimization. Infact, each question has been solved using some optimization routine involving `cvxopt.solvers.qp`.

## Files
The `Assignment.pdf` file contains all the details about the requests of each question, that are then developed in their own file inside the `Code` folder.
Also, the `Final_report.pdf` file describes the approach followed when developing such questions, explaining the decisions taken and all the performance results of such implementations (both in time and precision).
The report also provides various graphs (like those analyzing under/over fitting possible occurence and the confusion matrixes) and comments about the development, including a final comparison between all the developed techiniques.
