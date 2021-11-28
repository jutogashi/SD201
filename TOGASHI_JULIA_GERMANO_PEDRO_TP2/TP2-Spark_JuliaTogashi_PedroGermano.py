# Databricks notebook source
# Creating an RDD and loading the text file of 'steve.txt' on it
text_file = sc.textFile("/FileStore/tables/steve.txt")
print (text_file.take(1))

# COMMAND ----------

# Using 'flatMap' to split the text into a list of words
words = text_file.flatMap(lambda line: line.split(" "))
for a in words.take(5):
  print (a)

# COMMAND ----------

# Creating a list of pairs, where the word is the key and 1 is the value
pairs = words.map(lambda s: (s, 1)) # '2' turned into '1' (Question 1)
for (word, number) in pairs.take(5):
  print (word, number)

# COMMAND ----------

# Counting the number of occurences of each word, by using 'reduceByKey'
counts = pairs.reduceByKey(lambda a, b: a + b)
for (word, count) in counts.take(5):
  print (word, count)

# COMMAND ----------

# Sorting the list of tuples by 'sortBy' and choosing the second term (Question 2)
ordered = counts.sortBy(lambda x: x[1], False)
for a in ordered.take(5):
  print (a)

# COMMAND ----------

# Sorting (descending order) words with largest number of occurrences among the words containing at least 5 characters (Question 3)
atLeast5 = ordered.filter(lambda x: len(x[0]) > 5)
for a in atLeast5.take(5):
  print (a)

# COMMAND ----------

# Question 4
# Counting the words from list of links as in the last exercises
edge_file = sc.textFile("/FileStore/tables/edgelist.txt")
words_edge = edge_file.flatMap(lambda line: line.split(" "))
pairs_edge = words_edge.map(lambda s: (s, 1))
counts_edge = pairs_edge.reduceByKey(lambda a, b: a + b)

# Separate the label names and indexes
ids_file = sc.textFile("/FileStore/tables/idslabels.txt")
lines_ids = ids_file.flatMap(lambda line: line.splitlines())
ids = lines_ids.flatMap(lambda line: line.split(" ", 1)).collect()

# Getting the number and the name of the labels
ids_number = []
for i in ids[::2]:
  ids_number.append(i)
ids_name = []
for i in ids[1::2]:
  ids_name.append(i)

# Dictionary for label_number and Label_name
dict_ids = dict(zip(ids_number, ids_name))

# Print the name of links with most occurrences
for a, b in counts_edge.sortBy(lambda x: x[1], False).take(10):
  print ('%s : %i occurrences' %(dict_ids[a], b - 1)) # The label name doesn't count, so we need to subtract - 1 of the total count
