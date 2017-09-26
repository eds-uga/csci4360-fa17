The full assignment PDF can be found in the GitHub repository: https://github.com/eds-uga/csci4360-fa17/blob/master/assignments/assignment1.pdf

Any questions should be posted in the #questions channel of the course Slack chat: https://eds-uga-csci4360.slack.com/

The data for this problem is drawn from the 20 Newsgroups data
set. The training and test sets each contain 200 documents, 100 from
comp.sys.ibm.pc.hardware (label 0) and 100 from comp.sys.mac.hardware
(label 1). Each document is represented as a vector of word
counts.

The data consists of four files: train.data, train.label, test.data
and test.label. The .data files contain word count matrices whose rows
correspond to document_ids and whose columns correspond to
word_ids. Each row of the .data files represents the number of times a
certain word appeared in a certain document, in the following three
column format:

<document_id> <word_id> <count>

The .label files simply list the class label for each document in
order. i.e., the first entry of train.label is the label for the first
document in train.data.

You are also given PARTIAL testing sets—-the complete set is on the AutoLab autograder. The testing sets provided are to give you an idea of how your classifier will perform on the full dataset.

NOTES: For logistic regression, training can take several minutes (due to
gradient descent rules).
