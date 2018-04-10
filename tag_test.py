#USAGE
#compile it with python code.py
#OPTIONS
#insert -n or -y as options to recompile strings in test set
from itertools import izip
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



def progress(count, total, status=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '%' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s -> %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

#NOTE: create an automata with unknow words and also the lexer words

#variables
#variables
train_IOB = "dataset/data/NLSPARQL.train.data"
test_IOB = "dataset/data/NLSPARQL.test.data"
train_feats = "dataset/data/NLSPARQL.train.feats.txt"
test_feats = "dataset/data/NLSPARQL.test.feats.txt"
train_complete = "train_complete.txt"
test_complete = "test_complete.txt"
automata = "automata.txt"
lexer_name = "lexer_IOB"

lexer = []
converter = []
labels = []
unigrams = defaultdict(int)
bigrams = defaultdict(int)
tags_list = []

#create the new file that contain IOB tags and grammar tags
output = open(train_complete, "w")
with open(train_IOB) as file1, open(train_feats) as file2:
	for x, y in izip(file1, file2):
		w1 = x.split()
		w2 = y.split()
		#get the value
		print_str = ""
		if len(w1)>0:
			print_str = w1[0]+"\t"+w1[1]+"--"+w2[1]+"\n"
		else:
			print_str = "\n"

		output.write(print_str)

output.close()

output2 = open(test_complete, "w")
with open(test_IOB) as file1, open(test_feats) as file2:
	for x, y in izip(file1, file2):
		w1 = x.split()
		w2 = y.split()
		print(w1)
		print(w2)
		#get the value
		print_str = ""
		if len(w1)>0:
			print_str = w1[0]+"\t"+w1[1]+"--"+w2[1]+"\n"
		else:
			print_str = "\n"

		output2.write(print_str)

output2.close()

total = 2
for i in xrange(2):
	progress(i, total, 'Lexer creation')

#creation of common use variables: lexer
#create the lexer starting from the normal train data and also the dictionary for automata conversion

with open(train_complete) as text:
	for line in text:
		words = line.split()
		#check if the line is not empty
		if(len(words) > 0):
			words[0] = words[0].replace("'", "%")	#remove to obtain the base analysis
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

#print the lexer to file 
lexer_IOB = open("lexer_IOB.lex", "w")
lexer_IOB_txt = open("lexer_IOB.txt", "w")

#insert epsilon
lexer_IOB.write("<eps> 0\n")
lexer_IOB_txt.write("<eps> 0\n")

counter = 1

for l in lexer:
	lexer_IOB.write(l+" "+str(counter)+"\n")
	lexer_IOB_txt.write(l+" "+str(counter)+"\n")
	counter += 1

lexer_IOB.write("<unk> "+str(counter))
lexer_IOB_txt.write("<unk> "+str(counter))
lexer_IOB.close()
lexer_IOB_txt.close()


progress(2, total, 'Fst creation')

#--- AUTOMATA SECTION ---

#probability calc
#we have unigrams and bigrams, so we ca easly compute the probability as 
probab = defaultdict(float)
for elem, val in bigrams.items():
	#they have the same value, so use only the counter and
	p = -math.log(val/float(unigrams[elem[0]]))
	if p<0 or p==0:
		p=1
	probab[elem] = p
#now calculate probab for all the <unk> words
labels = list(set(labels))
probability_unk = float(1)/len(labels)

auth = open(automata, "w")
for elem, val in probab.items():
	auth.write("0 0 "+elem[1]+" "+elem[0]+" "+str(val)+"\n")

for l in labels:
	auth.write("0 0 <unk> "+l+" "+str(probability_unk)+"\n")

auth.write("0")
auth.close()


#now the compilation step is required for the automata
filename = "automata"
os.system('fstcompile --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex '+filename+'.txt > '+filename+'.fst')
#os.system('fstprint --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex '+filename+'.fst')

#creation of tag file starting from test, read second col and save it inside a file, to train the ML algorithm
tag_file = open("tag_train.txt", "w")
with open(train_complete) as file:
	final_str = ""
	for line in file:
		w = line.split()
		if len(w)>1:
			final_str += w[1]+"\t"
		else:
			final_str += "\n"
			tag_file.write(final_str)
			final_str = ""

tag_file.close()


#extract test strings to pass them to the lm algorithm
test_strings = []
test_file = []
with open(test_complete) as file:
	tmp_str = ""
	for line in file:
		w = line.split()
		test_file.append(line)
		if len(w)>1:
			tmp_str += w[0]+"\t"
		else:
			tmp_str += "\n"
			tmp_str = tmp_str.replace("'", "%")	#to avoid single quote problem in bash commands
			test_strings.append(tmp_str)
			tmp_str = ""



#--- END OF COMMON PART ---

#creation of arrays for all the methods
methods = ["absolute", "katz", "kneser_ney", "presmoothed", "witten_bell"]
orders = [2, 3, 4]	#start from 2 because with unsmoothed order 1 we have a baseline accuracy value
res_file = open("results.txt", "w")
res_file.close()

#for unsmoothed algorithm is important to use only unigrams, so change the code to achieve this
#goal, so we can start with this model
print("\n")
print("---START UNSMOOTHED---")
baseline_method = "unsmoothed"
os.system("mkdir -p "+baseline_method)
os.system("mkdir -p "+baseline_method+"/"+str(1))
path = baseline_method+"/"+str(1)+"/"
#reset prev files
os.system("rm "+path+"result_auth.txt")
os.system("rm "+path+"toEval.txt")
#generate far for the method and order 
os.system("farcompilestrings --symbols="+lexer_name+".txt --unknown_symbol='<unk>' tag_train.txt > "+path+"tag_train.far")
os.system("ngramcount --order="+str(1)+" --require_symbols=false "+path+"tag_train.far > "+path+"pos.cnt")
os.system("ngrammake --method="+baseline_method+" "+path+"pos.cnt > "+path+"pos.lm")
#now is possible to get the var test_strings and iterate over it and generate the automata
counter = 0
total = len(test_strings)
for ts in test_strings:
	#generating fst for the string
	os.system("echo '"+ts+"' | farcompilestrings --symbols="+lexer_name+".txt --unknown_symbol='<unk>' --generate_keys=1 --keep_symbols | farextract --filename_suffix='.fst'")
	#os.system('fstprint --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex 1.fst')
	#now is possible to pass the automa to the last step
	os.system("fstcompose 1.fst automata.fst | fstcompose - "+path+"pos.lm | fstrmepsilon | fstshortestpath > final.fst")
	os.system('fstprint --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex final.fst >> '+path+'result_auth.txt')
	os.system("echo '%%' >> "+path+"result_auth.txt")
	counter += 1
	progress(counter, total, 'Unsmoothed calc...')
#remove files
os.system("rm 1.fst")
os.system("rm final.fst")

#at the end of all the phrases we can read the file with the results and generate the toEval.txt file
toEval = open(path+"toEval.txt", "w")
with open(path+"result_auth.txt") as file:
	tmp_str = []
	for line in file:
		w = line.split()
		if len(w)>2:
			#common line
			tmp_val = w[3].split("--")	#remove for normal analysis
			tmp_str.append(tmp_val[0])
		elif len(w)==2:
			tmp_str.append('\n')
		elif w[0]=="%%":
			#now reorder -> first line, then reverse order
			print_str = tmp_str[0]+"\n"	#error here for out of range
			reverse_list = tmp_str[1:]
			reverse_list.reverse()
			for rl in reverse_list:
				if rl != "\n":
					print_str += rl+"\n"
				else:
					print_str += rl
			toEval.write(print_str)
			print_str = ""
			tmp_str = []

toEval.close()
#now concat files to obtain the final file with tags and predictions
os.system("paste dataset/data/NLSPARQL.test.data "+path+"toEval.txt > "+path+"final_pred.txt")
#evaluate the produced result
os.system("echo '"+baseline_method+" & order="+str(1)+"' >> results.txt")
os.system("./conlleval.pl -d '\t' < "+path+"final_pred.txt >> results.txt")	
#clean the result file

#---OTHER METHODS ---
print("\n")
#create a double for to see all the methods with all orders
for method in methods:
	os.system("mkdir -p "+method)
	print("\n")
	for order in orders:
		print("\n")
		print("Create fst for "+method+" & order="+str(order))
		os.system("mkdir -p "+method+"/"+str(order))
		path = method+"/"+str(order)+"/"
		#reset prev files
		os.system("rm "+path+"result_auth.txt")
		os.system("rm "+path+"toEval.txt")
		#generate far for the method and order 
		os.system("farcompilestrings --symbols="+lexer_name+".txt --unknown_symbol='<unk>' tag_train.txt > "+path+"tag_train.far")
		os.system("ngramcount --order="+str(order)+" --require_symbols=false "+path+"tag_train.far > "+path+"pos.cnt")
		os.system("ngrammake --method="+method+" "+path+"pos.cnt > "+path+"pos.lm")
		#now is possible to get the var test_strings and iterate over it and generate the automata
		counter = 0
		for ts in test_strings:
			#generating fst for the string
			os.system("echo '"+ts+"' | farcompilestrings --symbols="+lexer_name+".txt --unknown_symbol='<unk>' --generate_keys=1 --keep_symbols | farextract --filename_suffix='.fst'")
			#os.system('fstprint --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex 1.fst')
			#now is possible to pass the automa to the last step
			os.system("fstcompose 1.fst automata.fst | fstcompose - "+path+"pos.lm | fstrmepsilon | fstshortestpath > final.fst")
			os.system('fstprint --isymbols='+lexer_name+'.lex --osymbols='+lexer_name+'.lex final.fst >> '+path+'result_auth.txt')
			os.system("echo '%%' >> "+path+"result_auth.txt")
			#remove files
			os.system("rm 1.fst")
			os.system("rm final.fst")
			
			counter += 1
			progress(counter, total, method+" & order="+str(order)+" calc...")

		#at the end of all the phrases we can read the file with the results and generate the toEval.txt file
		toEval = open(path+"toEval.txt", "w")
		with open(path+"result_auth.txt") as file:
			tmp_str = []
			for line in file:
				w = line.split()
				if len(w)>2:
					#common line
					tmp_val = w[3].split("--")
					tmp_str.append(tmp_val[0])
				elif len(w)==2:
							tmp_str.append('\n')
				elif w[0]=="%%":
					#now reorder -> first line, then reverse order
					print_str = tmp_str[0]+"\n"
					reverse_list = tmp_str[1:]
					reverse_list.reverse()
					for rl in reverse_list:
						if rl != "\n":
							print_str += rl+"\n"
						else:
							print_str += rl

					toEval.write(print_str)
					print_str = ""
					tmp_str = []

		toEval.close()
		#now concat files to obtain the final file with tags and predictions
		os.system("paste dataset/data/NLSPARQL.test.data "+path+"toEval.txt > "+path+"final_pred.txt")
		#evaluate the produced result
		os.system("echo '"+method+" & order="+str(order)+"' >> results.txt")
		os.system("./conlleval.pl -d '\t' < "+path+"final_pred.txt >> results.txt")		
print("\n")
print("---DONE---")


			


