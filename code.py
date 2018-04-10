#USAGE
#compile it with python code.py
#OPTIONS
#insert -n or -y as options to recompile strings in test set


import string
import os
import numpy as np
from nltk import ngrams
import nltk
import math
from collections import Counter
from collections import defaultdict
import time
import sys

#TODO

#insert a control param to decide if recalc all the fst for each
#string

#insert arguments to the python script to tune the parameters

#purge the file creation inside fst function to avoid printing of the fst

#--- END TODO ---

#function to create the fst starting from automata and lexer
def createFST(filename, lexer):
	os.system('fstcompile --isymbols='+lexer+'.lex --osymbols='+lexer+'.lex '+filename+' > '+filename+'.fst')
	os.system('fstprint --isymbols='+lexer+'.lex --osymbols='+lexer+'.lex '+filename+'.fst')
	os.system('fstdraw --isymbols='+lexer+'.lex --osymbols='+lexer+'.lex '+filename+'.fst | dot -Tpng > '+filename+'.png')

def unionFST(fst1, fst2, out_fst):
	os.system("fstunion "+fst1+" "+fst2+" > "+out_fst+"")

def createTags(tags, lexer_txt):
	os.system("farcompilestrings --symbols="+lexer_txt+" --unknown_symbol='<unk>' "+tags+" > tag_file.far")
	os.system("ngramcount --order=3 --require_symbols=false tag_file.far > pos.cnt")
	os.system("ngrammake --method=witten_bell pos.cnt > pos.lm")

def compileStr(string, lexer_txt, lexer_lex):
	string = string.replace("'", "%")
	os.system("echo '"+string+"' | farcompilestrings --symbols="+lexer_txt+" --unknown_symbol='<unk>' --generate_keys=1 --keep_symbols | farextract --filename_suffix='.fst'")
	os.system('fstprint --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' 1.fst')
	os.system('fstdraw --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' 1.fst | dot -Tpng > 1.png')

	os.system("fstcompose union.fst 1.fst > c.fst")
	os.system("fstinvert c.fst > c2.fst")
	os.system("fstrmepsilon c2.fst > c3.fst")
	os.system("fstshortestpath c3.fst > compose.fst")
	os.system('fstprint --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' compose.fst')
	os.system('fstdraw --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' compose.fst | dot -Tpng > compose.png')
	

	#now is necessary using the lm files in order to obtain only the tags and save them inside a file
	
	os.system("fstcompose 1.fst compose.fst > medium.fst")
	os.system("fstcompose medium.fst pos.lm > cf2.fst")
	os.system("fstrmepsilon cf2.fst > cf22.fst")
	os.system("fstshortestpath cf22.fst > final.fst")
	os.system('fstprint --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' final.fst >> temp.txt')
	os.system('fstdraw --isymbols='+lexer_lex+' --osymbols='+lexer_lex+' final.fst | dot -Tpng > final.png')
	os.system('echo "%%" >> temp.txt')
	os.system('rm 1.fst')
	os.system('rm c.fst')
	os.system('rm c2.fst')
	os.system('rm c3.fst')
	os.system('rm compose.fst')

	os.system('rm medium.fst')
	os.system('rm cf2.fst')
	os.system('rm cf22.fst')
	os.system('rm final.fst')

#user variables
recompile = ""
if sys.argv[1]=="":
	recompile = "-y"
else:
	recompile = sys.argv[1]


#variables
train_IOB = "dataset/data/NLSPARQL.train.data"
test_IOB = "dataset/data/NLSPARQL.test.data"
automata = "automata.txt"
lexer_name = "lexer_IOB"

lexer = []
converter = []
labels = []
unigrams = defaultdict(int)
bigrams = defaultdict(int)
tags_list = []

#create the lexer starting from the normal train data and also the dictionary for automata conversion
with open(train_IOB) as text:
	for line in text:
		words = line.split()
		#check if the line is not empty
		if(len(words) > 0):
			words[0] = words[0].replace("'", "%")
			lexer.append(words[0])
			lexer.append(words[1])
			converter.append(words)
			unigrams[words[1]] += 1
			bigrams[words[1], words[0]] += 1
			labels.append(words[1])
			tags_list.append(words[1])

#convert lexer to a set to eliminate duplicates
lexer = list(set(lexer))

#convert the converter array into a dictionary in order to use it for automata
converter = dict(converter)

#--- LEXER SECTION ---

#print the lexer to file 
lexer_IOB = open("lexer_IOB.lex", "w")
lexer_IOB_txt = open("lexer_IOB.txt", "w")

#insert epsilon
lexer_IOB.write("<eps> 0\n")

counter = 1

for l in lexer:
	lexer_IOB.write(l+" "+str(counter)+"\n")
	lexer_IOB_txt.write(l+" "+str(counter)+"\n")
	counter += 1

lexer_IOB.write("<unk> "+str(counter))
lexer_IOB_txt.write("<unk> "+str(counter))
lexer_IOB.close()
lexer_IOB_txt.close()

#--- AUTOMATA SECTION ---

#we have unigrams and bigrams, so we ca easly compute the probability as 
probab = defaultdict(float)
for elem, val in bigrams.items():
	p = -math.log(val/float(unigrams[elem[0]]))
	probab[elem] = p


#now we have the probability, so we can create the automata as 0 0 A B probab
auth = open(automata, "w")
for elem, val in probab.items():
	auth.write("0 0 "+elem[0]+" "+elem[1]+" "+str(val)+"\n")

auth.write("0")
auth.close()

#now is possible to create the automata
createFST(automata, lexer_name)

#now we need to take in account unknow words, using the three possible labels observing that each of 
#them have a equal probability (1/3) so we can build a lexer and an automata for this. Using labels object

probability_unk = float(1) / len(labels)
unknow_lexer = open("unknow_lexer.lex", "w")
unknow_lexer_txt = open("unknow_lexer.txt", "w")
unknow_automata = open("unknow_automata.txt", "w")
counter = 0
labels = list(set(labels))
for l in labels:
	unknow_lexer.write(l+" "+str(counter)+"\n")
	unknow_lexer.write(l+" "+str(counter)+"\n")
	unknow_automata.write("0 0 <unk> "+l+" "+str(probability_unk)+"\n")
	counter += 1

#insert the final state
unknow_lexer.write("<unk> "+str(counter))
unknow_lexer_txt.write("<unk> "+str(counter))
unknow_automata.write("0")
unknow_lexer.close()
unknow_lexer_txt.close()
unknow_automata.close()

#create the fst for the unknow words automata
createFST("unknow_automata.txt", "unknow_lexer")

#now we can create a union of this two automata in order to create the final transducer. It is 
#called union.fst

print("---PERFORM fstunion---")
unionFST("automata.txt.fst", "unknow_automata.txt.fst", "union.fst")
print("--- DONE ---")


#automata with probabilities created, now we have to convert each phrase in test with automata and
#use < tmp.txt to save the result

#now we can read each phrase from the file and save it inside a list of string
phrase_str = []
tmp_str = ""
with open(test_IOB) as text:
	for line in text:
		w = line.split()
		if line=='\n':
			#new line only, so end of phrase
			tmp_str += "\n"
			phrase_str.append(tmp_str)
			tmp_str = ""
		else:
			#phrase not finished, so add the word
			tmp_str += w[0]+" " #append only the original word


#now take each phrase and convert into an fst and merge it with the union.fst to obtain predicted
#labels. Save the result for each phrase inside a tmp.txt file by append each predicted label
#discover

#we can use the object tags_list to obtain all tags, so we only need to save it inside a tag_file.txt
#that contains all the transduced strings in a compact format
tag_file = open("tags.txt", "w")
tmp_string = ""
list_string = []
with open(train_IOB) as file:
	for line in file:
		if line=='\n':
			list_string.append(tmp_string)
			tag_file.write(tmp_string+"\n")
			tmp_string = ""
		else:
			w = line.split()
			if(len(w)>0):
				tmp_string += w[1]+" "

#create tags files
createTags("tags.txt", lexer_name+".txt")


#Use that for testing purposes
res = []
#compileStr("actor from lost", lexer_name+".txt", lexer_name+".lex", res)


#use all possible strings in test and save the result to a file that contains the conversion

#now get all the automata for each phrase, that represents the conversion 

if recompile=="-y":
	reset_file = open("temp.txt", "w")
	reset_file.close()
	for s in phrase_str:
		compileStr(s, lexer_name+".txt", lexer_name+".lex")

#now we need to read the temp.txt file that contains all the automata. Read and store the result in a
#separated file contains only the third column and next order each automata label in the format: first
#row, others in reverse order
labels_file = open("final_label.txt", "w")
labels_array = []
prev = ""
with open("temp.txt") as file:
	for line in file:
		words = line.split()
		if(len(words)>2):
			labels_file.write(words[3]+"\n")
			prev = words[3]
		else:
			if words[0]=="%%" and prev!=words[0]:
				labels_file.write("%%\n")
				prev = words[0]
			elif words[0]=="":
				labels_file.write("\n")
				prev = ""

#now the list contains all the phrases separated by a %%
labels_file.close()
#read file
time.sleep(0.1)
labels_file = open("final_label.txt", "r")
with open("final_label.txt") as file:
	for line in file:
		labels_array.append(line)

to_eval = open("evaluate.txt", "w")
counter = 0
print(len(labels_array))	#not the same number
with open(test_IOB) as file:
	for line in file:
		w = line.split()
		tmp_str = ""
		final_str = ""
		if(len(w) > 1):
			final_str = ""+w[0]+"\t"+w[1]+"\t"+labels_array[counter]
		else:
			final_str = "\n"

		to_eval.write(final_str)
		counter += 1

to_eval.close()



