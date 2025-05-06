# csci567-program-3-hidden-markov-models-solved
**TO GET THIS SOLUTION VISIT:** [CSCI567 Program 3-Hidden Markov Models Solved](https://www.ankitcodinghub.com/product/csci567-program-3-hidden-markov-models-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;96572&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCI567 Program 3-Hidden Markov Models Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="section">
<div class="layoutArea">
<div class="column">
Hidden Markov Models (100 points)

</div>
</div>
<div class="layoutArea">
<div class="column">
In this programming assignment, we will implement Hidden Markov Models (HMM) and apply HMM to Part-of-Speech Tagging problem and Gene-tagging problem.

You need to first implement 6 important func!ons that help us accomplish various steps involved in learning HMM. Then you will be calcula!ng HMM parameters from the data and use it to solve Part-of- Speech Tagging problem and Gene-tagging problem.

A”er finishing the implementa!on, you can use hmm_test_script.py to test the correctness of your func!ons.

</div>
</div>
<div class="layoutArea">
<div class="column">
Mee”ng Link

h#ps://usc.zoom.us/j/5288651362 h#ps://usc.zoom.us/j/6625239116 h#ps://usc.zoom.us/j/98558703974 h#ps://usc.zoom.us/j/5288651362 h#ps://usc.zoom.us/j/6625239116

h#ps://usc.zoom.us/j/98558703974

</div>
<div class="column">
CP

Sowmya Anamay Amulya Sowmya Anamay

Amulya

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol>
<li>forward func!on – 10 = 5×2 points</li>
<li>backward func!on – 10 = 5×2 points</li>
<li>sequence_prob func!on – 5 = 5×1 points</li>
<li>posterior_prob func!on – 10 = 5×2 points</li>
<li>likelihood_prob func!on – 10 = 5×2 points</li>
<li>viterbi func!on – 15 = 5*3 points</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
There are 5 sets of grading data. Each grading data includes paramaters (pi, A, B, obs_dict, state_dict, Osequence) , which we will use to ini!alize HMM class and test your func!ons. To receive full credits, your output of func!on 1-5 should within an error of 10−8, your output of viterbi func!on should be iden!cal with ours.

For the defini!ons of the parameters pi , A , B etc., please refer to the code documenta!on. 1.2 Applica!on to Part-of-Speech Tagging

<ol>
<li>model_training func!on – 10 = 10x(your_correct_pred_cnt/our_correct_pred_cnt)</li>
<li>sentence_tagging func!on – 10 = 10x(your_correct_pred_cnt/our_correct_pred_cnt)</li>
</ol>
We will use the dataset given to you for grading this part (with a different random seed). We will train your model and our model on same train_data. model_training func!on and sentence_tagging func!on will be tested seperately.

In order to check your model_training func!on, we will use 50 sentences from train_data to do Part-of- Speech Tagging. To receive full credits, your predic!on accuracy should be iden!cal or be#er than ours.

In order to check your sentence_tagging func!on, we will use 50 sentences from test_data to do Part-of-Speech Tagging. To receive full credits, your predic!on accuracy should be iden!cal or be#er than ours.

1.3 Applica!on to Gene Tagging

<ol>
<li>model_training func!on – 10 = 10*(your_correct_pred_cnt/our_correct_pred_cnt)</li>
<li>sentence_tagging func!on – 10 = 10*(your_correct_pred_cnt/our_correct_pred_cnt)</li>
</ol>
Evaluated similar to the previous applica!on.

What to submit

hmm.py tagger.py

1.1 Implementa”on (60 points)

In 1.1, you are given parameters of a HMM and you will implement two procedures.

<ol>
<li>The Evalua”on Problem : Given HMM Model and a sequence of observa!ons, what is the probability that the observa!ons are generated by the model?
Two algorithms are usually used for the evalua!on problem: forward algorithm or the backward algorithm. Based on the result of forward algorithm and backward algorithm, you will be asked to calculate probability of sequence and posterior probability of state.
</li>
<li>The Decoding Problem : Given a model and a sequence of observa!ons, what is the most likely state sequence in the model which produced the observa!on sequence. For decoding you will be implemen!ng Viterbi algorithm.</li>
</ol>
HMM Class

In this project, we abstracted Hidden Markov Model as a class. Each Hiddern Markov Model ini!alized with Pi, A, B, obs_dict and state_dict . HMM class has 6 inner func!ons: forward func!on, backward

func!on, sequence_prob func!on, posterior_prob func!on, likelihood_prob and viterbi func!on.

<pre> ###
 You can add your own
</pre>
<pre> function or variables in HMM class, but you shouldn 't change current existing api. ###
 class HMM:
</pre>
def __init__(self, pi, A, B, obs_dict, state_dict):

-pi: (1 * num_state) A numpy array of initial probailities.pi[i] = P(X_1 = s_i) –

A: (num_state * num_state) A numpy array of transition probailities.A[i, j] = P(X_t = s_j | X_t – 1 = s_i) –

B: (num_state * num_obs_symbol) A numpy array of observation probabilities.B[i, o] = P(Z_t = z_o | X_t = s_i) – obs_dict: A dictionary mapping each observation symbol to their index in B –

state_dict: A dictionary mapping each state to their index in pi and A# TODO:

def forward(self, Osequence): #TODO:

def backward(self, Osequence): #TODO:

def sequence_prob(self, Osequence): #TODO:

def posterior_prob(self, Osequence): #TODO:

def likelihood_prob(self, Osequence): #TODO:

def viterbi(self, Osequence):

1.1.1 Evalua”on problem

(a) Forward algorithm and backward algorithm (20 points)

Here λ means the model. Please finish the implementa!on of forward() func!on and backward() func!on in hmm.py:

α[i, t] = P (Xt = si, Z1:t∣λ).

<pre> def forward(self, Osequence):
     ""
</pre>
” Inputs:

-self.pi: (1 * num_state) A numpy array of initial probailities.pi[i] = P(X_1 = s_i) –

self.A: (num_state * num_state) A numpy array of transition probailities.A[i, j] = P(X_t = s_j | X_t – 1 = s_i) – self.B: (num_state * num_obs_symbol) A numpy array of observation probabilities.B[i, o] = P(Z_t = z_o | X_t = s_i) – Osequence: (1 * L) A numpy array of observation sequence with length L

Returns:

-alpha: (num_state * L) A numpy array alpha[i, t] = P(X_t = s_i, Z_1: Z_t | λ)

“” ”

<pre> def backward(self, Osequence):
     ""
</pre>
” Inputs:

-self.pi: (1 * num_state) A numpy array of initial probailities.pi[i] = P(X_1 = s_i) –

self.A: (num_state * num_state) A numpy array of transition probailities.A[i, j] = P(X_t = s_j | X_t – 1 = s_i) – self.B: (num_state * num_obs_symbol) A numpy array of observation probabilities.B[i, o] = P(Z_t = z_o | X_t = s_i) – Osequence: (1 * L) A numpy array of observation sequence with length L

Returns:

-beta: (num_state * L) A numpy array beta[i, t] = P(Z_t + 1: Z_T | X_t = s_i, λ)

“” ”

(b) Sequence probability (5 points)

Based on your forward and backward func!on, you will calculate the sequence probability. (You can call forward func!on or backward func!on inside of sequence_prob func!on)

NN P(Z1,…,ZT = O∣λ) = ∑P(Xt = si,Z1:T∣λ) = ∑α[i,T]

</div>
</div>
<div class="layoutArea">
<div class="column">
β[i, t] = P (Zt+1:T ∣Xt = si, λ).

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>def sequence_prob(self, Osequence):
    ""
</pre>
” Inputs:

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1 i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>     -Osequence: (1 * L) A numpy array of observation sequence with length L
</pre>
Returns:

-prob: A float number of P(Z_1: Z_T | λ)

“” ”

© Posterior probability (10 points)

The forward variable α[i, t] and backward variable β[i, t] are used to calculate the posterior probability of a specific state. Now for t = 1…T and i = 1…N, we define posterior probability γt(i) = P(Xt = si∣O, λ) the probability of being in state si at !me t given the observa!on sequence O and the model λ.

γt(i) = P(Xt = si,O∣λ) = P(Xt = si,Z1:t∣λ) P (O∣λ) P (O∣λ)

P(Xt =si,Z1:t∣λ)=P(Z1:t∣Xt =si,λ)⋅P(Zt+1:T∣Xt =si,λ)⋅P(Xt =si∣λ)=α[i,t]⋅β[i,t] Thus

</div>
</div>
<div class="layoutArea">
<div class="column">
γt(i) = α[i, t] ⋅ β[i, t] P (O∣λ)

N P(O∣λ) = ∑α[i,T]

i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
where

Signature:

<pre> def posterior_prob(self, Osequence):
     ""
</pre>
” Inputs:

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>     -Osequence: (1 * L) A numpy array of observation sequence with length L
</pre>
Returns:

-prob: (num_state * L) A numpy array of P(X_t = i | O, λ)

“” ”

You can use βt(i) to find the most likely state at !me t which is the state Zt = si for which βt(i) is maximum. This algorithm works fine in the case when HMM is ergodic i.e. there is transi!on from any state to any other state. If applied to an HMM of another architecture, this approach could give a sequence that may not be a legi!mate path because some transi!ons are not permi#ed. To avoid this problem Viterbi algorithm is the most common decoding algorithms used.

(d) Likelihood of two consecu”ve states at a given “me (10 points)

You are required to calculate the likelihood of transi!on from state s at !me t to state s′ at !me t + 1. That is, you’re required to calculate

ξs,s′(t)=P(Xt =s,Xt+1 =s′ ∣Z1:T =z1:T)

Signature:

<pre> def likelihood_prob(self, Osequence):
     ""
</pre>
” Inputs:

<pre>     -Osequence: (1 * L) A numpy array of observation sequence with length L
</pre>
Returns:

-prob: (num_state * num_state * (L – 1)) A numpy array of P(X_t = i, X_t + 1 = j | O, λ)

“” ”

(e) Viterbi algorithm (15 points)

Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states. We want to compute the most likely state path that corresponds to the observa!on sequence O based HMM. Namely, k∗ = (k1∗,k2∗,⋅⋅⋅,kT∗ ) = argmaxk P(sk1,sk2,⋅⋅⋅,skT ∣Z1,Z2,⋅⋅⋅,ZT = O,λ).

Signature:

<pre> def viterbi(self, Osequence):
     ""
</pre>
” Inputs:

<pre>     -Osequence: (1 * L) A numpy array of observation sequence with length L
</pre>
<pre> Returns:
     -path: A List of the most likely hidden state path k * (
</pre>
<pre>         return state instead of idx)
</pre>
“” ”

1.2 Applica”on to Speech Tagging ( 20 points)

Part-of-Speech (POS) is a category of words (or, more generally, of lexical items) which have similar gramma!cal proper!es. (Example: noun, verb, adjec!ve, adverb, pronoun, preposi!on, conjunc!on, interjec!on, and some!mes numeral, ar!cle, or determiner.)

Part-of-Speech Tagging (POST) is the process of marking up a word in a text (corpus) as corresponding to a par!cular part of speech, based on both its defini!on and its context.

Here you will use HMM to do POST. You will need to calculate the parameters (pi, A, B, obs_dict, state_dict) of HMM first and then apply Viterbi algorithm to do speech-tagging.

Dataset

tags.txt: Universal Part-of-Speech Tagset

</div>
</div>
<div class="layoutArea">
<div class="column">
Tag Meaning

ADJ adjec!ve

ADP adposi!on

ADV adverb

CONJ conjunc!on

DET determiner, ar!cle NOUN noun

NUM numeral PRT par!cle PRON pronoun VERB verb

. punctua!on marks

X other

sentences.txt: Includes nearly 50000 sentences which have already been tagged.

Word Tag

b100-48585

She PRON had VERB

to PRT move VERB

in ADP some DET

direc!on NOUN –.

any DET direc!on NOUN

that PRON would VERB take VERB

her PRON away ADV from ADP

this DET

evil ADJ place NOUN

..

Part-of-Speech Tagging

In this part, we collect our dataset and tags with Dataset class. Dataset class includes tags, train_data and test_data. In both dataset include a list of sentences, each sentence is an object of Line class.

You only need to implement model_training func!on and sentence_tagging func!on. We have provided the accuracy func!on, which you can use to compare your predict_tagging and

true_tagging of a sentence. You can find the defini!on below.

###

You can add your own functions or variables in Dataset class, but you shouldn ‘t change current functions that exist. ## class Dataset:

<pre>     def __init__(self, tagfile, datafile, train_test_split = 0.8, seed = 112890):
</pre>
<pre>     self.tags
 self.train_data
</pre>
<pre> self.test_data
 def read_data(self, filename):
</pre>
<pre>     def read_tags(self, filename):
     class Line:
     def __init__(self, line, type):
     self.id
</pre>
<pre> self.words
 self.tags
 self.length
 def show(self): #TODO:
</pre>
<pre>     def model_training(train_data, tags)# TODO:
</pre>
<pre>     def sentence_tagging(model, test_data, tags)
 def accuracy(predict_tagging, true_tagging)
</pre>
1.2.1 Model training (10 points)

In this part, you will need to calculate the parameters of HMM model based on train_data . Signature:

<pre> def model_training(train_data, tags):
     ""
</pre>
” Inputs:

<pre>     -train_data: a list of sentences, each sentence is an object of Line class -
     tags: a list of POS tags
</pre>
Returns:

-model: an object of HMM class initialized with paramaters(pi, A, B, obs_dict, state_dict) you calculated based on t “”

”

1.2.2 Speech_tagging (10 points)

Based on HMM from 1.2.1, do speech tagging for each sentence on test data. Note when you meet a word which is unseen in training dataset. You need to modify the emission matrix and obs_dict of your current model in order to handle this case. You will assume the emission probability from each state to a new unseen word is 10−6(a very low probability).

<pre> For example, in hmm_model.json, we use the following paramaters to initialize HMM:
     S = ["1", "2"]
</pre>
<pre> pi: [0.7, 0.3]
 A: [
</pre>
[0.8, 0.2],

[0.4, 0.6] ]

B=[

[0.5, 0, 0.4, 0.1], [0.5, 0.1, 0.2, 0.2]

]

Observations = [“A”, “C”, “G”, “T”]

If we find another observation symbol “X” in observation sequence, we will modify parameters of HMM as follows:

<pre>     S = ["1", "2"]
 pi: [0.7, 0.3]
</pre>
<pre> A: [
     [0.8, 0.2],
</pre>
[0.4, 0.6] ]

B=[

[0.5, 0, 0.4, 0.1, 1e-6], [0.5, 0.1, 0.2, 0.2, 1e-6]

<pre> ]
 Observations = ["A", "C", "G", "T", "X"]
</pre>
You do not get access to test_data on model_training func!on, you need to implement the logic to tag a new sequence in sentence_tagging func!on.

Signature:

<pre> def sentence_tagging(test_data, model):
     ""
</pre>
” Inputs:

<pre>     -test_data: (1 * num_sentence) a list of sentences, each sentence is an object of Line class -
     model: an object of HMM class
</pre>
<pre> Returns:
     -tagging: (num_sentence * num_tagging) a 2 D list of output tagging
</pre>
<pre> for each sentences on test_data
     ""
</pre>
”

1.2.3 Sugges”on(0 points)

This part won’t be graded. In order to have a be#er understanding of HMM. Come up with one sentence by yourself and tagging it manually. Then run your forward func!on, backward func!on, seq_prob func!on, posterior_prob func!on and viterbi func!on on the model from 1.2.1. Print the result of each func!on, see if you can explain your result.

1.3 Applica”on to Gene Tagging (20 points)

In this task you’ll use the HMM class to perform Gene tagging in sentences.

This applica!on is similar to the previous POS Tagging applica!on, the difference is that you’ll tag gene names in biological text data. You will use use an HMM for this task. You will need to calculate the parameters (pi, A, B, obs_dict, state_dict) of HMM first and then apply Viterbi algorithm to do gene-tagging.

Dataset

gene_tags.txt: Tagset for the given gene dataset

</div>
</div>
<div class="layoutArea">
<div class="column">
English Examples

new, good, high, special, big, local on, of, at, with, by, into, under really, already, s!ll, early, now

and, or, but, if, while, although

the, a, some, most, every, no, which year, home, costs, !me, Africa twenty-four, fourth, 1991, 14:24 at, on, out, over per, that, up, with he, their, her, its, my, I, us

is, say, told, given, playing, would . , ;!

</div>
</div>
<div class="layoutArea">
<div class="column">
ersatz, esprit, dunno, gr8, univeristy

</div>
</div>
<div class="layoutArea">
<div class="column">
Tag Meaning

GENE Gene names

O Other

genes.txt: Includes about 3000 sentences which have already been tagged.

Word Tag

Serum GENE gamma GENE glutamyltransferase GENE

in O the O diagnosis O of O liver O disease O in O ca#le O

.O

Likelihood values

Use the likelihood values of consecu!ve states computed for a sentence to understand how the model is able to recognize gene names split into mul!ple words.

Note

Do not be deceived by the high tagging accuracy in this task. While our model can recognize the genes in most of the cases, even a model that tags every word as O would have a high accuracy since it’s much more likely to find an O than a GENE. This is one of the classic examples where accuracy is not the right way to evaluate the model. Try some examples from the dataset to see this for yourself.

</div>
</div>
<div class="layoutArea">
<div class="column">
Examples

5-neuclio!dase, carbonic anhydrase etc.

</div>
</div>
<div class="layoutArea">
<div class="column">
Any word that’s not a gene

</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
#

r

</div>
</div>
</div>
</div>
