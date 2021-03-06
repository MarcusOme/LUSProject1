# Sequence Labelling using Finite State Transducers
University of Trento - Mid Term Project

## Required tools
Some tools are required in order to run the code presented in this project.

* [Python 2.7](https://www.python.org/downloads/): installed and configured properly
* [OpenFst](http://www.openfst.org/): library for constructing, combining, optimizing and searching weighted finite-state transducers(FSTs)
* [OpenGrm](http://www.openfst.org/twiki/bin/view/GRM/NGramLibrary): collection of open-source libraries for constructing, combining, applying and searching formal grammars and creating n-grams models as FSTs

## Download and execute the code

It is possible to clone and enter the GitHub repository with the following command:

```
git clone https:https://github.com/MarcusOme/LUSProject1.git
cd LUSProject1
```
After cloning the repository is possible execute one of the .py files inside without any parameters. For example:

```
python tag_test.py
```

Each python file perform a different operation based on train and test set manipulation:

* base.py - use the IOB tags present in train and create a simple prediction model for the test set.
* cutoff.py - perform cutoff of order 1 in train IOB labels
* O_excl.py - delete O tags from train to perform analysis
* tag_test.py - merge tags from IOB and Lemmas
* IOB_exclusion.py - delete O, I-movie.name and B-movie.name tags. Best performance script.

The result of the execution can be seen inside the *results.txt*. The file contains a list of all the methods and n-grams order tried and the correlated accuracy. **Beware: this file will be override every time a .py script is executed.**

## Project Folders

### /dataset

Contains all datasets in the subfolder **/data**. The files with extension .data contain the IOB tags, instead the ones with feats.txt extension contains POS-Tags. For more details see the report.

### /Final_results

Already contains the result calculated for each python script in the project. Each file contains the results of different methods with different n-gram orders. **Those files will not be override while running code, so you can compare the result obtained with the ones already present.**

### /absolute & /katz & /kneser_ney /presmoothed & /unsmoothed & /witten_bell

Those folders contains .lm files generated by python code and have different subfolder (from 2 to 4 except for unsmoothed) in which LM for each method and n-gram order are saved. **In case of delete simply run the script to generate those files again.**

## Other files

In the home folder are present other files. Those are used and generated by scripts to execute and should not be modified.

## Author

**Marco Omezzolli** - marco.omezzolli@studenti.unitn.it
