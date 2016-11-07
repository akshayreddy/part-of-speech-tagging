###################################
# CS B551 Fall 2016, Assignment #3
#
# Akshay reddyak@iu.edu
#
# (Based on skeleton code by D. Crandall)
#
#
####
# -*- coding: ascii -*-
"""
Description of problem formulation, approach and programming methodologies

structure=Dictionary
By state we mean parts of speech

First Subsection:

In this section we had to use the training data and compute the Initial state probabilities, Transition 
probabilities and emission probabilities.

Sol:-

Initial state probabilities:

1.) To calculate the Initial state probabilities we created a function named Initial_State_Probability.
2.) The function creates an empty structure (IWP) which keeps on incrementing when same initial state is 
    found and sets the initial count to 1 if that State is not present in the structure.
3.) Now the structure contained values of 12 states which was then divided by number of sentences in the 
    training data in order to get the initial state probabilities.
4.) Since the training data had yield probabilities of all the 12 states, no further smoothing is required.

Transition probabilities:

1.) To calculate the Transition probabilities we created a function named Transition_Probability.
2.) The function created an empty structure (GAPOS); we traversed from second to last state of the sentence
    in order to get the count of number of same states given same previous states. Which was later divided number
    of number of states to get the probabilities, i.e.,
    CountOf(noun|verb)/ NumberOf(noun).
3.) The training data yield 143 such combination which was 1 less to all the combinations of Transition 
    probabilities; which is later set to the lowest probability amongst all combinations but greater than zero.

Emission probabilities:

1.) To calculate the Transition probabilities we created a function named Emission_Probability.
2.) The function created an empty structure (GAW); we traversed both the sentence and label list 
    parallelly to get the count of number of word having same state. Which is later divided by the the number 
    of counts of given word to get the probabilities.
    CountOf(word|noun,verb...)/ NumberOf(word).
3.) The training data yield many combination which sometimes may not be sufficient for a give test data. 
    That is the word or the combination may not be present in the trained data. In future such combinations 
    are given the least probabilities amongst all emission probabilities but greater than zero.



Second Subsection (simplified version):
In this section we had to label the given sentences with parts of speech using the above calculated 
probabilities and given model 1(b)
Sol:-

1.) Looking at the model we concluded that no hidden states have any relationships amongst themselves
    except that each of them having one observable state.
2.) The function (Simplified)finds the maximum marginal probabilities of each word in the sentence given 
    all the parts of speech which was in this case to find the maximum emission probabilities for each word 
    given all the parts of speech.
3.) We multiplied the Initial state probability with the emission probability when calculating the state for
    the first word.


Third Subsection (HMM version):
In this section we had to implement the viterbi algorithm and deduce the maximum a posteriori labeling for a 
given sentence using model(1a).

1.) Model 1(a) is a perfect HMM model. To implement the viterbi algorithms we initially created structure (v) 
    with labels containing all the 12 parts of speech.
2.) The resulting structure will contain a 12xT matrix form of values with row symbolizing parts of speech 
    and T representing the number of words in sentence. Column T0, T1...Tn is virtually considered to be time 
    stamps where one's values are used during the upcoming timestamps.
3.) Initial time stamp (T0) is calculated just by multiplying the respective initial probabilities and 
    emission probabilities all across the row. Later for future time stamp(n) transition probabilities 
    and T(n-1) is multiple all across the row respectively and the maximum amongst that list is multiplied to 
    the emission probability.
.i.e., max([v(n-1)(noun,verb...)*TransitionP(noun|(noun,verb...))])*EmissionP(word|noun)

4.) In the end to find the maximum a posteriori the structure (v) is traversed coloum wise to get the 
maximum value which indeed gives the respective state.

Forth Subsection (Complex version):
In this section we had to label the given sentence with the parts of speech by implementing the 
variable elimination method on model (1c).

1.) For this we created a structure (t) with labels containing all the 12 parts of speech.
2.) Similar to the HMM model T0 involved multiplying the respective initial probabilities and emission 
probabilities all across the row.
3.)But for T1 is calculated by summing the respective initial probabilities and emission probabilities 
all across the row. This in involves a summation of 12 values.
4.) Form T2 till Tn same procedure as step 3 is applied but this involves summation of 144 values across 
the row.
.i.e., sum([t(n-1)(noun,verb...)*TransitionP(noun|(noun,verb...)+t(n-2)(noun,verb...)*TransitionP(noun|(noun,verb...))])*EmissionP(word|noun)

5.) At last the sequence is deduced by finding the maximum values column wise which is finding the 
maximum marginal probabilities for each words.


Logarithm of posterior probabilities:
We have calculated the posterior probabilities based on the model(1a). Which is, the functions returns
the summation of log of initial probability, log of transition probabilities and log of emission probabilities.

Assumptions:
1.) The transition probabilities which are not yield from the training data are set with a probability 
    of 0.01e-6, (in the HMM fuction). Since we know least information about this value we set it to the lowest
    value amongst all. Since setting a probability to zero will tamper the future calculations, the value is greater than zero.
2.) Similarly the emission probabilities which are not yield from the training data are set with a 
    probability of 0.01e-6 and assigned a POS 'X' (Foreign Key) since the words is out of the context of the training data.


Result and analysis:

The program successfully runs over the given input file (bc.test).
After running over 2000 sentences with 29422 words it give the result as follows,

Simplified: Word correctness (80-83%),Sentence Correctness(16.40%)
HMM: Word correctness (76-79%), Sentence Correctness(14.40%)
Complex:Word correctness (77-79%), Sentence Correctness(14.40%)

Thought :
We were analysising the output to understand why the HMM and Complex have a low word correctness 
than simplified. We looked an some sentences one of which was sentence no. 20 from bc.test.The HMM and 
Complex model was able to tag the word 'his' with 'pronoun' whereas Ground truth and simplified tagged 
it as determiner. In grammatical sense pronouns are subsets of determiner.By this we can see HMM and 
Complex due to the transition probabilities were able to tag the POS specifically whereas simplified 
followed the generic approach. We know this is out of scope of the assignment, just a thought.

"""





####

import random
import math
import collections
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling

    Traind_IWP=[]  #Initial_State_Probability
    Traind_GAP=[]  #Transition_Probability
    Traind_GAW=[]  #Emission_Probability

    count=0
    def posterior(self, sentence, label):
        prob=0
        for i in range(len(sentence)):
            index="%s|%s"%(sentence[i],label[i])
            if index not in Solver.Traind_GAW[0]:
                Solver.Traind_GAW[0][index]=0.1e-6

        prob=prob+math.log(Solver.Traind_IWP[label[0]])+sum([math.log(Solver.Traind_GAW[0]["%s|%s"%(sentence[x],label[x])]) for x in range(len(sentence))])
        prob=prob+sum([math.log(Solver.Traind_GAP[0]["%s|%s" % (label[x],label[x-1])]) for x in range(1,len(label))])

        return prob

    # Do the training!
    #

    def Initial_State_Probability(self,data):                 #P(S_1)
        IWP={}
        Sentence_count=0

        for i in data:
            temp="%s" % (i[1][0])
            if temp in IWP:
                IWP[temp]=IWP[temp]+1
            else:
                IWP[temp]=1
            Sentence_count+=1

        for i in IWP:
            IWP[i]=IWP[i]/float(Sentence_count)
        return IWP

    def Transition_Probability(self,data):                               #P(S_n+1|S_n)
        POS={}
        GAPOS={}
        for i in data:
            for j in range(len(i[1])-1):
                if len(i[1])==1:
                    continue

                temp2="%s" % (i[1][j+1])
                if temp2 in POS:
                    POS[temp2]=POS[temp2]+1
                else:
                    POS[temp2]=1

        for i in data:
            for j in range(len(i[1])-1):
                temp1="%s|%s" % (i[1][j+1],i[1][j])
                if temp1 in GAPOS:
                    GAPOS[temp1]=(GAPOS[temp1]+1)/float(POS[i[1][j+1]])
                else:
                    GAPOS[temp1]=1/float(POS[i[1][j+1]])
        return [GAPOS,POS]

    def Emission_Probability(self,data):                              #P(W|S)
        Word_Count={}
        GAW={}
        for i in data:
            for j in range(len(i[1])):
                temp2="%s" % (i[0][j])
                if temp2 in Word_Count:
                    Word_Count[temp2]=Word_Count[temp2]+1
                else:
                    Word_Count[temp2]=1

        for i in data:
            for j in range(len(i[1])):
                temp1="%s|%s" % (i[0][j],i[1][j])
                if temp1 in GAW:
                    GAW[temp1]=(GAW[temp1]+1)/float(Word_Count[i[0][j]])
                else:
                    GAW[temp1]=1/float(Word_Count[i[0][j]])
        return [GAW,Word_Count]


    def train(self, data):
        Solver.Traind_IWP=self.Initial_State_Probability(data)
        Solver.Traind_GAP=self.Transition_Probability(data)
        Solver.Traind_GAW=self.Emission_Probability(data)

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        POS_Pridictions_1=[]
        Word_Probability=[]
        Types=['adv','noun','adp','prt','det','num','x','pron','verb','.','conj','adj']

        for i in sentence:
            maximum=0.01e-3
            temp_POS=''
            for j in Types:
                if i not in Solver.Traind_GAW[1]:
                    Solver.Traind_GAW[0]["%s|%s" % (i,'x')]=0.1e-6
                    temp_POS='x'
                temp="%s|%s" % (i,j)
                if temp in Solver.Traind_GAW[0]:
                    if Solver.Traind_GAW[0][temp]>maximum:
                        maximum=Solver.Traind_GAW[0][temp]
                        temp_POS=j
            if i==0:
                Word_Probability.append(maximum*Solver.Traind_IWP[temp_POS])
            else:
                Word_Probability.append(maximum)
            POS_Pridictions_1.append(temp_POS)

        return [ [POS_Pridictions_1], [Word_Probability] ]
        

    def hmm(self, sentence):
        POS_Pridictions_2=[]
        v={}
        Types=['adv','noun','adp','prt','det','num','x','pron','verb','.','conj','adj']
        for i in Types:
            v[i]=[1 for j in range(len(sentence))]

        #Smoothening 
        for i in Types:
            for j in Types:
                temp="%s|%s" % (i,j)
                if temp not in Solver.Traind_GAP[0]:
                    Solver.Traind_GAP[0][temp]=0.01e-6

        for i in range(len(sentence)):
            if i==0:
                for j in Types:
                    index="%s|%s"%(sentence[i],j)
                    if index in Solver.Traind_GAW[0]:
                        v[j][i]=((Solver.Traind_IWP[j])*(Solver.Traind_GAW[0][index]))
                    else:
                        Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]=0.1e-6
                        v[j][i]=((Solver.Traind_IWP[j])*(Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]))
            else:
                for j in Types:
                    index="%s|%s"%(sentence[i],j)
                    if index in Solver.Traind_GAW[0]:
                        v[j][i]=max([v[x][i-1]*Solver.Traind_GAP[0]["%s|%s"%(j,x)] for x in Types])*Solver.Traind_GAW[0][index]
                    else:
                        Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]=0.1e-6
                        v[j][i]=max([v[x][i-1]*Solver.Traind_GAP[0]["%s|%s"%(j,x)] for x in Types])*Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]


        for i in range(len(sentence)):
            temp=[[v[x][i],x] for x in Types]
            maximum=max([v[x][i] for x in Types])
            for j in temp:
                if j[0]==maximum:
                    POS_Pridictions_2.append(j[1])

        return [[POS_Pridictions_2[0:len(sentence)]],[]]



    def complex(self, sentence):
        POS_Pridictions_3=[]
        Word_Probability=[]
        t={}

        Types=['adv','noun','adp','prt','det','num','x','pron','verb','.','conj','adj']
        for i in Types:
            t[i]=[1 for j in range(len(sentence))]

        for i in range(len(sentence)):
            if i==0:
                for j in Types:
                    index="%s|%s" % (sentence[i],j)
                    if index in Solver.Traind_GAW[0]:
                        t[j][i]=Solver.Traind_IWP[j]*Solver.Traind_GAW[0][index]
                    else:
                        Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]=0.1e-6
                        t[j][i]=Solver.Traind_IWP[j]*Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]
            elif i==1:
                for j in Types:
                    index="%s|%s" % (sentence[i],j)
                    if index in Solver.Traind_GAW[0]:
                        t[j][i]=sum([Solver.Traind_GAP[0]["%s|%s" % (j,x)]*t[x][i-1] for x in Types])*Solver.Traind_GAW[0][index]
                    else:
                        Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]=0.1e-6
                        t[j][i]=sum([Solver.Traind_GAP[0]["%s|%s" % (j,x)]*t[x][i-1] for x in Types])*Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]
            else:
                for j in Types:
                    index="%s|%s" % (sentence[i],j)
                    if index in Solver.Traind_GAW[0]:
                        t[j][i]=sum([ Solver.Traind_GAP[0]["%s|%s" % (j,Types[x])]*t[Types[x]][i-1]+[Solver.Traind_GAP[0]["%s|%s" % (j,Types[y])]*t[Types[y]][i-2] for y in range(len(Types))][x] for x in range(len(Types))])*Solver.Traind_GAW[0][index]
                    else:
                        Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]=0.1e-6
                        t[j][i]=sum([ Solver.Traind_GAP[0]["%s|%s" % (j,Types[x])]*t[Types[x]][i-1]+[Solver.Traind_GAP[0]["%s|%s" % (j,Types[y])]*t[Types[y]][i-2] for y in range(len(Types))][x] for x in range(len(Types))])*Solver.Traind_GAW[0]["%s|%s" % (sentence[i],'x')]

        for i in range(len(sentence)):
            temp=[[t[x][i],x] for x in Types]
            maximum=max([t[x][i] for x in Types])
            for j in temp:
                if j[0]==maximum:
                    Word_Probability.append(j[0])
                    POS_Pridictions_3.append(j[1])

        return [ [POS_Pridictions_3[0:len(sentence)] ], [Word_Probability[0:len(sentence)]] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

